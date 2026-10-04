from __future__ import annotations

import json

import duckdb
import joblib
import numpy as np
import pandas as pd

try:
    from .runtime_context import (
        load_runtime_context_from_env,
    )
    from .forecast_evaluation_context import (
        build_forecast_evaluation_context,
    )
    from .train_forecast_benchmarks import (
        prepare_x as prepare_benchmark_x,
        make_elastic_net_pipeline,
        make_hist_gradient_boosting_pipeline,
    )
    from .train_sparse_forecast_models import (
        prepare_x as prepare_sparse_x,
        make_poisson_pipeline,
        make_occurrence_pipeline,
        make_positive_amount_pipeline,
    )
    from .train_quantile_forecasts import (
        prepare_x as prepare_quantile_x,
        make_quantile_pipeline,
        QUANTILES,
    )
except ImportError:
    from runtime_context import (
        load_runtime_context_from_env,
    )
    from forecast_evaluation_context import (
        build_forecast_evaluation_context,
    )
    from train_forecast_benchmarks import (
        prepare_x as prepare_benchmark_x,
        make_elastic_net_pipeline,
        make_hist_gradient_boosting_pipeline,
    )
    from train_sparse_forecast_models import (
        prepare_x as prepare_sparse_x,
        make_poisson_pipeline,
        make_occurrence_pipeline,
        make_positive_amount_pipeline,
    )
    from train_quantile_forecasts import (
        prepare_x as prepare_quantile_x,
        make_quantile_pipeline,
        QUANTILES,
    )


RUNTIME_CONTEXT = load_runtime_context_from_env()

EVALUATION_CONTEXT = (
    build_forecast_evaluation_context(
        RUNTIME_CONTEXT
    )
)

DATABASE_PATH = (
    RUNTIME_CONTEXT.database_path
)

TARGET_YEAR = int(
    RUNTIME_CONTEXT.planning_year
)

FEATURE_CUTOFF_YEAR = int(
    RUNTIME_CONTEXT.latest_completed_year
)

COMBINED_OOS_LABEL = (
    EVALUATION_CONTEXT.combined_oos_label
)

FORECAST_VERSION = (
    f"FORECAST_{TARGET_YEAR}_"
    "GOVERNED_CANDIDATE_V1"
)

OUTPUT_DIR = (
    RUNTIME_CONTEXT.processed_dir
    / "forecasting"
    / (
        f"forecast_{TARGET_YEAR}_"
        "governed_candidate_v1"
    )
)

MODEL_DIR = (
    RUNTIME_CONTEXT.models_dir
    / "forecasting"
    / (
        f"forecast_{TARGET_YEAR}_"
        "governed_candidate_v1"
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCORING_TABLE = (
    f"forecasting_scoring_{TARGET_YEAR}"
)

VALID_POINT_MODELS = {
    "NAIVE_LAG1",
    "MOVING_AVERAGE_2",
    "ELASTIC_NET_GLOBAL",
    "HIST_GRADIENT_BOOSTING_GLOBAL",
    "HIST_GRADIENT_BOOSTING_POISSON",
    "HURDLE_HGB_EXPECTED",
}


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        supervised = connection.execute(
            """
            SELECT *
            FROM forecasting_supervised
            ORDER BY Target_Year, Medicine_Key
            """
        ).df()

        scoring = connection.execute(
            f"""
            SELECT *
            FROM {SCORING_TABLE}
            ORDER BY Medicine_Key
            """
        ).df()

        model_governance = (
            connection.execute(
                """
                SELECT *
                FROM forecast_model_selection_governance
                """
            ).df()
        )

        quantile_validation = (
            connection.execute(
                """
                SELECT *
                FROM forecast_quantile_metrics_overall
                WHERE Fold_Name = ?
                ORDER BY Nominal_Quantile
                """,
                [COMBINED_OOS_LABEL],
            ).df()
        )

        calibration = connection.execute(
            """
            SELECT *
            FROM forecast_quantile_calibration_comparison
            ORDER BY Nominal_Quantile
            """
        ).df()

    finally:
        connection.close()

    return (
        supervised,
        scoring,
        model_governance,
        quantile_validation,
        calibration,
    )


def validate_inputs(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
    governance: pd.DataFrame,
    quantile_validation: pd.DataFrame,
    calibration: pd.DataFrame,
) -> str:
    if supervised.empty:
        raise ValueError(
            "Supervised forecasting dataset is empty."
        )

    expected_years = set(
        RUNTIME_CONTEXT.supervised_target_years
    )

    actual_years = set(
        supervised[
            "Target_Year"
        ].unique().tolist()
    )

    if actual_years != expected_years:
        raise ValueError(
            "Unexpected supervised target years. "
            f"Expected {sorted(expected_years)}; "
            f"found {sorted(actual_years)}."
        )

    if scoring.empty:
        raise ValueError(
            "Planning-year scoring dataset is empty."
        )

    if scoring[
        "Medicine_Key"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Medicine_Key in scoring data."
        )

    if not scoring[
        "Target_Year"
    ].eq(
        TARGET_YEAR
    ).all():
        raise ValueError(
            "Unexpected scoring Target_Year."
        )

    if not scoring[
        "Feature_Cutoff_Year"
    ].eq(
        FEATURE_CUTOFF_YEAR
    ).all():
        raise ValueError(
            "Unexpected scoring Feature_Cutoff_Year."
        )

    if not scoring[
        "Max_History_Year_Used"
    ].eq(
        FEATURE_CUTOFF_YEAR
    ).all():
        raise ValueError(
            "Scoring data contains unexpected "
            "history-year evidence."
        )

    if len(governance) != 1:
        raise ValueError(
            "Expected exactly one current "
            "forecast governance decision."
        )

    governance_row = (
        governance.iloc[0]
    )

    if int(
        governance_row[
            "Planning_Year"
        ]
    ) != TARGET_YEAR:
        raise ValueError(
            "Model governance Planning_Year "
            "does not match candidate target."
        )

    if (
        governance_row[
            "Evaluation_Population"
        ]
        != COMBINED_OOS_LABEL
    ):
        raise ValueError(
            "Model governance evidence window "
            "does not match current evaluation."
        )

    recommended_model = str(
        governance_row[
            "Recommended_Model"
        ]
    )

    if (
        recommended_model
        not in VALID_POINT_MODELS
    ):
        raise ValueError(
            "Unsupported governed point model: "
            f"{recommended_model}"
        )

    if set(
        quantile_validation[
            "Quantile_Label"
        ].tolist()
    ) != set(
        QUANTILES
    ):
        raise ValueError(
            "Combined quantile validation is incomplete."
        )

    if set(
        calibration[
            "Quantile_Label"
        ].tolist()
    ) != set(
        QUANTILES
    ):
        raise ValueError(
            "Temporal calibration comparison "
            "is incomplete."
        )

    return recommended_model


def train_hurdle(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
):
    x_train = prepare_sparse_x(
        supervised
    )

    x_score = prepare_sparse_x(
        scoring
    )

    occurrence_target = (
        supervised[
            "Target_Occurrence"
        ]
        .astype(int)
        .to_numpy()
    )

    occurrence_model = (
        make_occurrence_pipeline()
    )

    occurrence_model.fit(
        x_train,
        occurrence_target,
    )

    probability = (
        occurrence_model.predict_proba(
            x_score
        )[:, 1]
    )

    probability = np.clip(
        probability,
        0,
        1,
    )

    positive_train = (
        supervised.loc[
            supervised[
                "Target_Annual_Issuance"
            ].gt(0)
        ]
        .copy()
    )

    if len(
        positive_train
    ) < 100:
        raise ValueError(
            "Too few positive training rows "
            "for hurdle amount model."
        )

    positive_x = prepare_sparse_x(
        positive_train
    )

    positive_y_log = np.log1p(
        positive_train[
            "Target_Annual_Issuance"
        ].to_numpy(
            dtype=float
        )
    )

    amount_model = (
        make_positive_amount_pipeline()
    )

    amount_model.fit(
        positive_x,
        positive_y_log,
    )

    conditional_log = (
        amount_model.predict(
            x_score
        )
    )

    conditional_quantity = np.clip(
        np.expm1(
            conditional_log
        ),
        0,
        None,
    )

    hurdle_forecast = (
        probability
        *
        conditional_quantity
    )

    return (
        probability,
        conditional_quantity,
        hurdle_forecast,
        occurrence_model,
        amount_model,
    )


def score_point_model(
    model_name: str,
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
    hurdle_forecast: np.ndarray,
):
    if model_name == "NAIVE_LAG1":
        prediction = pd.to_numeric(
            scoring[
                "Lag1_Issuance"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        model = None

    elif model_name == "MOVING_AVERAGE_2":
        ma2_prediction = pd.to_numeric(
            scoring[
                "Moving_Average_2"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        lag1_fallback = pd.to_numeric(
            scoring[
                "Lag1_Issuance"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        fallback_mask = (
            ~np.isfinite(
                ma2_prediction
            )
        )

        if (
            fallback_mask.any()
            and
            not np.isfinite(
                lag1_fallback[
                    fallback_mask
                ]
            ).all()
        ):
            raise ValueError(
                "MOVING_AVERAGE_2 fallback "
                "requires valid Lag1_Issuance."
            )

        prediction = np.where(
            fallback_mask,
            lag1_fallback,
            ma2_prediction,
        )

        model = None

    elif model_name in {
        "ELASTIC_NET_GLOBAL",
        "HIST_GRADIENT_BOOSTING_GLOBAL",
    }:
        if (
            model_name
            == "ELASTIC_NET_GLOBAL"
        ):
            model = (
                make_elastic_net_pipeline()
            )
        else:
            model = (
                make_hist_gradient_boosting_pipeline()
            )

        x_train = prepare_benchmark_x(
            supervised
        )

        x_score = prepare_benchmark_x(
            scoring
        )

        y_train = supervised[
            "Target_Annual_Issuance"
        ].to_numpy(
            dtype=float
        )

        model.fit(
            x_train,
            np.log1p(
                y_train
            ),
        )

        prediction = np.expm1(
            model.predict(
                x_score
            )
        )

    elif (
        model_name
        == "HIST_GRADIENT_BOOSTING_POISSON"
    ):
        model = (
            make_poisson_pipeline()
        )

        x_train = prepare_sparse_x(
            supervised
        )

        x_score = prepare_sparse_x(
            scoring
        )

        y_train = supervised[
            "Target_Annual_Issuance"
        ].to_numpy(
            dtype=float
        )

        model.fit(
            x_train,
            y_train,
        )

        prediction = model.predict(
            x_score
        )

    elif (
        model_name
        == "HURDLE_HGB_EXPECTED"
    ):
        prediction = (
            hurdle_forecast.copy()
        )

        model = None

    else:
        raise ValueError(
            f"Unknown point model: {model_name}"
        )

    prediction = np.clip(
        np.asarray(
            prediction,
            dtype=float,
        ),
        0,
        None,
    )

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            f"{model_name} produced "
            "missing/non-finite predictions."
        )

    return (
        prediction,
        model,
    )


def train_quantiles(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
):
    x_train = prepare_quantile_x(
        supervised
    )

    x_score = prepare_quantile_x(
        scoring
    )

    y_train = supervised[
        "Target_Annual_Issuance"
    ].to_numpy(
        dtype=float
    )

    predictions = {}
    models = {}

    for label, quantile in (
        QUANTILES.items()
    ):
        model = (
            make_quantile_pipeline(
                quantile
            )
        )

        model.fit(
            x_train,
            y_train,
        )

        prediction = np.clip(
            model.predict(
                x_score
            ),
            0,
            None,
        )

        if not np.isfinite(
            prediction
        ).all():
            raise ValueError(
                f"{label} produced "
                "non-finite forecasts."
            )

        predictions[
            label
        ] = prediction

        models[
            label
        ] = model

    return (
        predictions,
        models,
    )


def calibration_review_flag(
    calibration: pd.DataFrame,
) -> bool:
    planning_rows = (
        calibration.loc[
            calibration[
                "Quantile_Label"
            ].isin(
                [
                    "P80",
                    "P90",
                ]
            )
        ]
    )

    return bool(
        len(planning_rows) == 2
        and
        planning_rows[
            "Calibration_Improved_Coverage_Error"
        ].all()
        and
        planning_rows[
            "Calibration_Improved_Pinball"
        ].all()
    )


def build_package(
    scoring: pd.DataFrame,
    point_prediction: np.ndarray,
    point_model: str,
    occurrence_probability: np.ndarray,
    conditional_positive: np.ndarray,
    hurdle_forecast: np.ndarray,
    quantile_predictions: dict,
    governance: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    result = scoring[
        [
            "Medicine_Key",
            "Brand_Name",
            "Generic_Name",
            "Unit",
            "Target_Year",
            "Feature_Cutoff_Year",
            "ABC_AsOf",
            "Behavior_Segment_AsOf",
            "Lag1_Issuance",
            "Lag2_Issuance",
            "Moving_Average_2",
            "Persistence_AsOf",
            "Positive_Years_AsOf",
            "Zero_Years_AsOf",
            "Lag1_Ending_Balance",
            "Lag1_Valid_Average_Cost",
            "Lag1_Cost_Valid",
            "Lag1_Data_Quality_Status",
        ]
    ].copy()

    result[
        "Point_Forecast"
    ] = point_prediction

    result[
        "Point_Forecast_Model"
    ] = point_model

    result[
        "Point_Forecast_Status"
    ] = (
        "GOVERNANCE_RECOMMENDED_CANDIDATE"
    )

    if (
        point_model
        == "MOVING_AVERAGE_2"
    ):
        ma2_values = pd.to_numeric(
            scoring[
                "Moving_Average_2"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        point_fallback = (
            ~np.isfinite(
                ma2_values
            )
        )

    else:
        point_fallback = np.zeros(
            len(result),
            dtype=bool,
        )

    result[
        "Point_Forecast_Fallback_Applied"
    ] = point_fallback

    result[
        "Point_Forecast_Effective_Model"
    ] = np.where(
        point_fallback,
        "NAIVE_LAG1",
        point_model,
    )

    result[
        "Point_Forecast_Fallback_Reason"
    ] = np.where(
        point_fallback,
        (
            "INSUFFICIENT_HISTORY_FOR_"
            "MOVING_AVERAGE_2"
        ),
        "",
    )

    result[
        "Hurdle_Occurrence_Probability"
    ] = occurrence_probability

    result[
        "Hurdle_Conditional_Positive_Quantity"
    ] = conditional_positive

    result[
        "Hurdle_Point_Forecast_Reference"
    ] = hurdle_forecast

    result[
        "Hurdle_Model_Status"
    ] = (
        "GOVERNANCE_RECOMMENDED_AND_"
        "OCCURRENCE_RISK"
        if point_model
        == "HURDLE_HGB_EXPECTED"
        else
        "OCCURRENCE_RISK_REFERENCE"
    )

    raw_matrix = np.column_stack(
        [
            quantile_predictions["P50"],
            quantile_predictions["P80"],
            quantile_predictions["P90"],
        ]
    )

    adjusted = np.maximum.accumulate(
        raw_matrix,
        axis=1,
    )

    result[
        "Raw_P50"
    ] = raw_matrix[:, 0]

    result[
        "Raw_P80"
    ] = raw_matrix[:, 1]

    result[
        "Raw_P90"
    ] = raw_matrix[:, 2]

    result[
        "P50"
    ] = adjusted[:, 0]

    result[
        "P80"
    ] = adjusted[:, 1]

    result[
        "P90"
    ] = adjusted[:, 2]

    result[
        "Raw_Quantile_Crossing"
    ] = (
        (
            raw_matrix[:, 0]
            >
            raw_matrix[:, 1]
        )
        |
        (
            raw_matrix[:, 1]
            >
            raw_matrix[:, 2]
        )
    )

    result[
        "Quantile_Adjustment_Applied"
    ] = (
        (
            raw_matrix
            != adjusted
        )
        .any(
            axis=1
        )
    )

    result[
        "P80_minus_P50"
    ] = (
        result["P80"]
        -
        result["P50"]
    )

    result[
        "P90_minus_P50"
    ] = (
        result["P90"]
        -
        result["P50"]
    )

    result[
        "P50_Use_Status"
    ] = (
        "STATISTICAL_OUTPUT_NOT_POINT_FORECAST"
    )

    result[
        "P80_Use_Status"
    ] = (
        "PLANNING_SCENARIO_NOT_SERVICE_GUARANTEE"
    )

    result[
        "P90_Use_Status"
    ] = (
        "PLANNING_SCENARIO_NOT_SERVICE_GUARANTEE"
    )

    eligible_review = (
        calibration_review_flag(
            calibration
        )
    )

    result[
        "Residual_Calibration_Applied"
    ] = False

    result[
        "Residual_Calibration_Eligible_For_Human_Review"
    ] = eligible_review

    result[
        "Uncertainty_Policy"
    ] = (
        "RAW_BACKTESTED_QUANTILES_"
        "NO_AUTOMATIC_RESIDUAL_RECALIBRATION"
    )

    governance_row = (
        governance.iloc[0]
    )

    result[
        "Model_Governance_Decision"
    ] = governance_row[
        "Decision"
    ]

    result[
        "Model_Governance_Reason"
    ] = governance_row[
        "Decision_Reason"
    ]

    result[
        "Model_Governance_Run_ID"
    ] = governance_row[
        "Run_ID"
    ]

    result[
        "Forecast_Version"
    ] = FORECAST_VERSION

    result[
        "Run_ID"
    ] = RUNTIME_CONTEXT.run_id

    result[
        "Run_Mode"
    ] = RUNTIME_CONTEXT.run_mode

    result[
        "Publication_Status"
    ] = "CANDIDATE_NOT_PUBLISHED"

    result[
        "Observed_Target_Issuance_Used"
    ] = False

    return result


def validate_package(
    result: pd.DataFrame,
    scoring: pd.DataFrame,
    recommended_model: str,
) -> None:
    if len(
        result
    ) != len(
        scoring
    ):
        raise ValueError(
            "Candidate package population "
            "does not match scoring population."
        )

    if result[
        "Medicine_Key"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Medicine_Key "
            "in candidate package."
        )

    required_numeric = [
        "Point_Forecast",
        "Hurdle_Occurrence_Probability",
        "Hurdle_Conditional_Positive_Quantity",
        "Hurdle_Point_Forecast_Reference",
        "P50",
        "P80",
        "P90",
    ]

    for column in required_numeric:
        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Missing/non-finite {column}."
            )

    quantity_columns = [
        "Point_Forecast",
        "Hurdle_Conditional_Positive_Quantity",
        "Hurdle_Point_Forecast_Reference",
        "P50",
        "P80",
        "P90",
    ]

    if result[
        quantity_columns
    ].lt(0).any().any():
        raise ValueError(
            "Negative forecast quantity."
        )

    if (
        result[
            "Hurdle_Occurrence_Probability"
        ].lt(0).any()
        or
        result[
            "Hurdle_Occurrence_Probability"
        ].gt(1).any()
    ):
        raise ValueError(
            "Occurrence probability "
            "outside [0,1]."
        )

    if result[
        "P50"
    ].gt(
        result["P80"]
    ).any():
        raise ValueError(
            "P50 > P80."
        )

    if result[
        "P80"
    ].gt(
        result["P90"]
    ).any():
        raise ValueError(
            "P80 > P90."
        )

    if not result[
        "Target_Year"
    ].eq(
        TARGET_YEAR
    ).all():
        raise ValueError(
            "Unexpected target year."
        )

    if not result[
        "Feature_Cutoff_Year"
    ].eq(
        FEATURE_CUTOFF_YEAR
    ).all():
        raise ValueError(
            "Unexpected feature cutoff year."
        )

    if not result[
        "Point_Forecast_Model"
    ].eq(
        recommended_model
    ).all():
        raise ValueError(
            "Point model does not match "
            "governance recommendation."
        )

    if (
        recommended_model
        == "MOVING_AVERAGE_2"
    ):
        expected_fallback = (
            ~np.isfinite(
                pd.to_numeric(
                    scoring[
                        "Moving_Average_2"
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
            )
        )
    else:
        expected_fallback = np.zeros(
            len(result),
            dtype=bool,
        )

    actual_fallback = result[
        "Point_Forecast_Fallback_Applied"
    ].to_numpy(
        dtype=bool
    )

    if not np.array_equal(
        actual_fallback,
        expected_fallback,
    ):
        raise ValueError(
            "Point forecast fallback flags "
            "do not match scoring evidence."
        )

    if (
        result.loc[
            actual_fallback,
            "Point_Forecast_Effective_Model",
        ]
        .ne("NAIVE_LAG1")
        .any()
    ):
        raise ValueError(
            "MA2 fallback did not record "
            "NAIVE_LAG1 as effective model."
        )

    if result[
        "Observed_Target_Issuance_Used"
    ].any():
        raise ValueError(
            "Outcome leakage flag "
            "unexpectedly TRUE."
        )

    if result[
        "Residual_Calibration_Applied"
    ].any():
        raise ValueError(
            "Residual calibration was "
            "automatically applied."
        )


def build_segment_summary(
    result: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for segment_type in [
        "ABC_AsOf",
        "Behavior_Segment_AsOf",
    ]:
        for segment, group in (
            result.groupby(
                segment_type,
                dropna=False,
                sort=True,
            )
        ):
            rows.append(
                {
                    "Segment_Type":
                        segment_type,
                    "Segment":
                        (
                            str(segment)
                            if pd.notna(segment)
                            else "MISSING"
                        ),
                    "Rows":
                        len(group),
                    "Point_Forecast_Total":
                        float(
                            group[
                                "Point_Forecast"
                            ].sum()
                        ),
                    "Hurdle_Reference_Total":
                        float(
                            group[
                                "Hurdle_Point_Forecast_Reference"
                            ].sum()
                        ),
                    "P50_Total":
                        float(
                            group["P50"].sum()
                        ),
                    "P80_Total":
                        float(
                            group["P80"].sum()
                        ),
                    "P90_Total":
                        float(
                            group["P90"].sum()
                        ),
                    "Mean_Occurrence_Probability":
                        float(
                            group[
                                "Hurdle_Occurrence_Probability"
                            ].mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_training_summary(
    supervised: pd.DataFrame,
    point_model: str,
) -> pd.DataFrame:
    train_start = int(
        supervised[
            "Target_Year"
        ].min()
    )

    train_end = int(
        supervised[
            "Target_Year"
        ].max()
    )

    positive_rows = int(
        supervised[
            "Target_Annual_Issuance"
        ].gt(0).sum()
    )

    point_training_rows = (
        0
        if point_model in {
            "NAIVE_LAG1",
            "MOVING_AVERAGE_2",
        }
        else
        len(supervised)
    )

    rows = [
        {
            "Component":
                point_model,
            "Role":
                "GOVERNED_POINT_FORECAST",
            "Training_Rows":
                point_training_rows,
            "Training_Target_Year_Start":
                (
                    None
                    if point_training_rows == 0
                    else train_start
                ),
            "Training_Target_Year_End":
                (
                    None
                    if point_training_rows == 0
                    else train_end
                ),
        },
        {
            "Component":
                "HURDLE_OCCURRENCE",
            "Role":
                "OCCURRENCE_RISK",
            "Training_Rows":
                len(supervised),
            "Training_Target_Year_Start":
                train_start,
            "Training_Target_Year_End":
                train_end,
        },
        {
            "Component":
                "HURDLE_POSITIVE_AMOUNT",
            "Role":
                "OCCURRENCE_RISK_QUANTITY_COMPONENT",
            "Training_Rows":
                positive_rows,
            "Training_Target_Year_Start":
                train_start,
            "Training_Target_Year_End":
                train_end,
        },
    ]

    for label in QUANTILES:
        rows.append(
            {
                "Component":
                    f"QUANTILE_{label}",
                "Role":
                    "UNCERTAINTY_MODEL",
                "Training_Rows":
                    len(supervised),
                "Training_Target_Year_Start":
                    train_start,
                "Training_Target_Year_End":
                    train_end,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_package_governance(
    model_governance: pd.DataFrame,
    quantile_validation: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    governance_row = (
        model_governance.iloc[0]
    )

    p80 = calibration.loc[
        calibration[
            "Quantile_Label"
        ].eq("P80")
    ].iloc[0]

    p90 = calibration.loc[
        calibration[
            "Quantile_Label"
        ].eq("P90")
    ].iloc[0]

    return pd.DataFrame(
        [
            {
                "Forecast_Version":
                    FORECAST_VERSION,
                "Run_ID":
                    RUNTIME_CONTEXT.run_id,
                "Run_Mode":
                    RUNTIME_CONTEXT.run_mode,
                "Planning_Year":
                    TARGET_YEAR,
                "Feature_Cutoff_Year":
                    FEATURE_CUTOFF_YEAR,
                "Evaluation_Population":
                    COMBINED_OOS_LABEL,
                "Governance_Decision":
                    governance_row[
                        "Decision"
                    ],
                "Governance_Reason":
                    governance_row[
                        "Decision_Reason"
                    ],
                "Recommended_Point_Model":
                    governance_row[
                        "Recommended_Model"
                    ],
                "Automatic_Model_Promotion_Allowed":
                    bool(
                        governance_row[
                            "Automatic_Promotion_Allowed"
                        ]
                    ),
                "P80_Calibration_Improved_Coverage_Error":
                    bool(
                        p80[
                            "Calibration_Improved_Coverage_Error"
                        ]
                    ),
                "P80_Calibration_Improved_Pinball":
                    bool(
                        p80[
                            "Calibration_Improved_Pinball"
                        ]
                    ),
                "P90_Calibration_Improved_Coverage_Error":
                    bool(
                        p90[
                            "Calibration_Improved_Coverage_Error"
                        ]
                    ),
                "P90_Calibration_Improved_Pinball":
                    bool(
                        p90[
                            "Calibration_Improved_Pinball"
                        ]
                    ),
                "Residual_Calibration_Eligible_For_Human_Review":
                    calibration_review_flag(
                        calibration
                    ),
                "Residual_Calibration_Applied":
                    False,
                "Publication_Status":
                    "CANDIDATE_NOT_PUBLISHED",
            }
        ]
    )


def save_models(
    point_model_name: str,
    point_model,
    occurrence_model,
    amount_model,
    quantile_models: dict,
) -> None:
    if point_model is not None:
        joblib.dump(
            {
                "model_name":
                    point_model_name,
                "pipeline":
                    point_model,
                "forecast_version":
                    FORECAST_VERSION,
                "training_through":
                    FEATURE_CUTOFF_YEAR,
                "target_year":
                    TARGET_YEAR,
            },
            MODEL_DIR
            / "governed_point_model.joblib",
        )

    joblib.dump(
        {
            "pipeline":
                occurrence_model,
            "component":
                "HURDLE_OCCURRENCE",
            "forecast_version":
                FORECAST_VERSION,
        },
        MODEL_DIR
        / "hurdle_occurrence.joblib",
    )

    joblib.dump(
        {
            "pipeline":
                amount_model,
            "component":
                "HURDLE_POSITIVE_AMOUNT",
            "forecast_version":
                FORECAST_VERSION,
        },
        MODEL_DIR
        / "hurdle_positive_amount.joblib",
    )

    for label, model in (
        quantile_models.items()
    ):
        joblib.dump(
            {
                "pipeline":
                    model,
                "quantile":
                    QUANTILES[
                        label
                    ],
                "forecast_version":
                    FORECAST_VERSION,
            },
            MODEL_DIR
            / (
                f"{label.lower()}_"
                f"{TARGET_YEAR}.joblib"
            ),
        )


def write_database(
    result: pd.DataFrame,
    segment_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    package_governance: pd.DataFrame,
) -> None:
    table_base = (
        f"forecast_{TARGET_YEAR}_"
        "governed_candidate"
    )

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:
        connection.register(
            "forecast_df",
            result,
        )

        connection.register(
            "segment_df",
            segment_summary,
        )

        connection.register(
            "training_df",
            training_summary,
        )

        connection.register(
            "package_governance_df",
            package_governance,
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {table_base} AS
            SELECT *
            FROM forecast_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {table_base}_segment_summary AS
            SELECT *
            FROM segment_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {table_base}_training_summary AS
            SELECT *
            FROM training_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {table_base}_governance AS
            SELECT *
            FROM package_governance_df
            """
        )

    finally:
        connection.close()


def main() -> None:
    print("=" * 108)
    print("CDH INVENTORY DSS")
    print(
        f"{TARGET_YEAR} GOVERNED "
        "CANDIDATE FORECAST PACKAGE"
    )
    print("=" * 108)
    print()

    (
        supervised,
        scoring,
        model_governance,
        quantile_validation,
        calibration,
    ) = load_inputs()

    recommended_model = (
        validate_inputs(
            supervised,
            scoring,
            model_governance,
            quantile_validation,
            calibration,
        )
    )

    print(
        "Input validation: PASS"
    )
    print(
        "Planning year:",
        TARGET_YEAR,
    )
    print(
        "Feature cutoff year:",
        FEATURE_CUTOFF_YEAR,
    )
    print(
        "Scoring population:",
        f"{len(scoring):,}",
    )
    print(
        "Governed point model:",
        recommended_model,
    )
    print()

    (
        occurrence_probability,
        conditional_positive,
        hurdle_forecast,
        occurrence_model,
        amount_model,
    ) = train_hurdle(
        supervised,
        scoring,
    )

    (
        point_prediction,
        point_model,
    ) = score_point_model(
        recommended_model,
        supervised,
        scoring,
        hurdle_forecast,
    )

    (
        quantile_predictions,
        quantile_models,
    ) = train_quantiles(
        supervised,
        scoring,
    )

    result = build_package(
        scoring=scoring,
        point_prediction=point_prediction,
        point_model=recommended_model,
        occurrence_probability=occurrence_probability,
        conditional_positive=conditional_positive,
        hurdle_forecast=hurdle_forecast,
        quantile_predictions=quantile_predictions,
        governance=model_governance,
        calibration=calibration,
    )

    validate_package(
        result,
        scoring,
        recommended_model,
    )

    print(
        "Candidate package structural validation: PASS"
    )

    segment_summary = (
        build_segment_summary(
            result
        )
    )

    training_summary = (
        build_training_summary(
            supervised,
            recommended_model,
        )
    )

    package_governance = (
        build_package_governance(
            model_governance,
            quantile_validation,
            calibration,
        )
    )

    save_models(
        point_model_name=recommended_model,
        point_model=point_model,
        occurrence_model=occurrence_model,
        amount_model=amount_model,
        quantile_models=quantile_models,
    )

    result.to_parquet(
        OUTPUT_DIR
        / (
            f"forecast_{TARGET_YEAR}_"
            "governed_candidate.parquet"
        ),
        index=False,
        engine="pyarrow",
    )

    result.to_csv(
        OUTPUT_DIR
        / (
            f"forecast_{TARGET_YEAR}_"
            "governed_candidate.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    segment_summary.to_csv(
        OUTPUT_DIR
        / (
            f"forecast_{TARGET_YEAR}_"
            "segment_summary.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    training_summary.to_csv(
        OUTPUT_DIR
        / (
            f"forecast_{TARGET_YEAR}_"
            "training_summary.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    package_governance.to_csv(
        OUTPUT_DIR
        / (
            f"forecast_{TARGET_YEAR}_"
            "governance.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "forecast_version":
            FORECAST_VERSION,
        "run_id":
            RUNTIME_CONTEXT.run_id,
        "run_mode":
            RUNTIME_CONTEXT.run_mode,
        "planning_year":
            TARGET_YEAR,
        "feature_cutoff_year":
            FEATURE_CUTOFF_YEAR,
        "evaluation_population":
            COMBINED_OOS_LABEL,
        "recommended_point_model":
            recommended_model,
        "publication_status":
            "CANDIDATE_NOT_PUBLISHED",
        "automatic_publication":
            False,
    }

    (
        OUTPUT_DIR
        / "package_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_database(
        result=result,
        segment_summary=segment_summary,
        training_summary=training_summary,
        package_governance=package_governance,
    )

    print()
    print(
        "Point forecast total:",
        f"{result['Point_Forecast'].sum():,.2f}",
    )
    print(
        "Hurdle reference total:",
        f"{result['Hurdle_Point_Forecast_Reference'].sum():,.2f}",
    )
    print(
        "P50 total:",
        f"{result['P50'].sum():,.2f}",
    )
    print(
        "P80 total:",
        f"{result['P80'].sum():,.2f}",
    )
    print(
        "P90 total:",
        f"{result['P90'].sum():,.2f}",
    )
    print(
        "Quantile-adjusted rows:",
        int(
            result[
                "Quantile_Adjustment_Applied"
            ].sum()
        ),
    )
    print(
        "Residual calibration applied:",
        False,
    )
    print(
        "Candidate automatically published:",
        False,
    )

    print()
    print("=" * 108)
    print(
        f"{TARGET_YEAR} GOVERNED "
        "CANDIDATE FORECAST COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
