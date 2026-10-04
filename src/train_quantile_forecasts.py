from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .runtime_context import (
        load_runtime_context_from_env,
    )
    from .forecast_evaluation_context import (
        build_forecast_evaluation_context,
    )
except ImportError:
    from runtime_context import (
        load_runtime_context_from_env,
    )
    from forecast_evaluation_context import (
        build_forecast_evaluation_context,
    )


# ============================================================
# PATHS
# ============================================================

RUNTIME_CONTEXT = (
    load_runtime_context_from_env()
)

EVALUATION_CONTEXT = (
    build_forecast_evaluation_context(
        RUNTIME_CONTEXT
    )
)

PROJECT_ROOT = (
    RUNTIME_CONTEXT.project_root
)

DATABASE_PATH = (
    RUNTIME_CONTEXT.database_path
)

OUTPUT_DIR = (
    RUNTIME_CONTEXT.processed_dir
    / "forecasting"
    / "quantile_v1"
)

MODEL_DIR = (
    RUNTIME_CONTEXT.models_dir
    / "forecasting"
    / "quantile_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

QUANTILE_VERSION = "FORECAST_QUANTILE_V1"
RANDOM_STATE = 42

QUANTILES = {
    "P50": 0.50,
    "P80": 0.80,
    "P90": 0.90,
}

FOLDS = EVALUATION_CONTEXT.folds

COMBINED_OOS_LABEL = (
    EVALUATION_CONTEXT.combined_oos_label
)


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "Lag1_Issuance",
    "Lag2_Issuance",
    "Moving_Average_2",
    "Historical_Mean_Issuance",
    "Historical_Median_Issuance",
    "Persistence_AsOf",
    "Mean_Positive_Issuance_AsOf",
    "CV_Positive_Issuance_AsOf",
    "Trend_Slope_AsOf",
    "Lag1_Ending_Balance",
    "Lag1_Deliveries",
    "Lag1_Total_In",
    "Lag1_Total_Out",
    "Lag1_Valid_Average_Cost",
]

CATEGORICAL_FEATURES = [
    "Unit",
    "ABC_AsOf",
    "Behavior_Segment_AsOf",
    "Lag1_Data_Quality_Status",
    "Lag1_Cost_Valid",
    "Lag1_Ending_Balance_Negative",
]

ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# LOAD DATA
# ============================================================

def load_supervised() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            """
            SELECT *
            FROM forecasting_supervised
            ORDER BY Target_Year, Medicine_Key
            """
        ).df()

    finally:

        connection.close()

    return dataframe


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_x(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    x = dataframe[
        ALL_FEATURES
    ].copy()

    for column in NUMERIC_FEATURES:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    for column in CATEGORICAL_FEATURES:

        x[column] = (
            x[column]
            .astype("object")
        )

        x[column] = (
            x[column]
            .where(
                pd.notna(
                    x[column]
                ),
                np.nan,
            )
        )

    return x


# ============================================================
# PREPROCESSING
# ============================================================

def make_preprocessor() -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="MISSING",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# QUANTILE MODEL
# ============================================================

def make_quantile_pipeline(
    quantile: float,
) -> Pipeline:

    estimator = (
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                make_preprocessor(),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# RUN ONE FOLD
# ============================================================

def run_fold(
    supervised: pd.DataFrame,
    fold: dict,
) -> tuple[
    pd.DataFrame,
    list[dict],
]:

    fold_name = fold[
        "Fold_Name"
    ]

    train_start = fold[
        "Train_Start"
    ]

    train_end = fold[
        "Train_End"
    ]

    test_year = fold[
        "Test_Year"
    ]

    train = (
        supervised.loc[
            supervised[
                "Target_Year"
            ]
            .between(
                train_start,
                train_end,
            )
        ]
        .copy()
    )

    test = (
        supervised.loc[
            supervised[
                "Target_Year"
            ]
            .eq(
                test_year
            )
        ]
        .copy()
    )

    if test.empty:

        raise ValueError(
            f"{fold_name}: no test rows found."
        )

    if test[
        "Medicine_Key"
    ].duplicated().any():

        raise ValueError(
            f"{fold_name}: duplicate "
            "Medicine_Key in test population."
        )

    x_train = prepare_x(
        train
    )

    x_test = prepare_x(
        test
    )

    y_train = (
        train[
            "Target_Annual_Issuance"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print()
    print(
        f"Running {fold_name}"
    )

    print(
        f"  Train target years: "
        f"{train_start}-{train_end}"
    )

    print(
        f"  Train rows: "
        f"{len(train):,}"
    )

    print(
        f"  Test year: "
        f"{test_year}"
    )

    raw_predictions = {}

    model_records = []

    for label, quantile in (
        QUANTILES.items()
    ):

        print(
            f"  Fitting quantile: "
            f"{label} ({quantile:.2f})"
        )

        pipeline = (
            make_quantile_pipeline(
                quantile
            )
        )

        pipeline.fit(
            x_train,
            y_train,
        )

        prediction = (
            pipeline.predict(
                x_test
            )
        )

        prediction = np.clip(
            prediction,
            0,
            None,
        )

        raw_predictions[
            label
        ] = prediction

        model_path = (
            MODEL_DIR
            / (
                f"{fold_name.lower()}"
                f"__{label.lower()}.joblib"
            )
        )

        joblib.dump(
            {
                "pipeline":
                    pipeline,

                "quantile":
                    quantile,

                "quantile_label":
                    label,

                "fold":
                    fold,

                "numeric_features":
                    NUMERIC_FEATURES,

                "categorical_features":
                    CATEGORICAL_FEATURES,

                "model_version":
                    QUANTILE_VERSION,
            },
            model_path,
        )

        model_records.append(
            {
                "Fold_Name":
                    fold_name,

                "Quantile_Label":
                    label,

                "Quantile":
                    quantile,

                "Train_Target_Year_Start":
                    train_start,

                "Train_Target_Year_End":
                    train_end,

                "Test_Target_Year":
                    test_year,

                "Train_Row_Count":
                    len(train),

                "Test_Row_Count":
                    len(test),

                "Model_Path":
                    str(
                        model_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "Quantile_Version":
                    QUANTILE_VERSION,
            }
        )

    # ========================================================
    # QUANTILE CROSSING BEFORE ADJUSTMENT
    # ========================================================

    raw_matrix = np.column_stack(
        [
            raw_predictions["P50"],
            raw_predictions["P80"],
            raw_predictions["P90"],
        ]
    )

    crossing_p50_p80 = (
        raw_matrix[:, 0]
        >
        raw_matrix[:, 1]
    )

    crossing_p80_p90 = (
        raw_matrix[:, 1]
        >
        raw_matrix[:, 2]
    )

    any_crossing = (
        crossing_p50_p80
        |
        crossing_p80_p90
    )

    # ========================================================
    # OPERATIONAL MONOTONICITY ADJUSTMENT
    #
    # Ensures:
    # P50 <= P80 <= P90
    #
    # This adjustment will be explicitly
    # reported and evaluated.
    # ========================================================

    adjusted = np.maximum.accumulate(
        raw_matrix,
        axis=1,
    )

    result = test[
        [
            "Medicine_Key",
            "Brand_Name",
            "Generic_Name",
            "Unit",
            "Target_Year",
            "Feature_Cutoff_Year",
            "ABC_AsOf",
            "Behavior_Segment_AsOf",
            "Target_Annual_Issuance",
            "Target_Occurrence",
        ]
    ].copy()

    result[
        "Fold_Name"
    ] = fold_name

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
        "Raw_P50_P80_Crossing"
    ] = crossing_p50_p80

    result[
        "Raw_P80_P90_Crossing"
    ] = crossing_p80_p90

    result[
        "Raw_Any_Quantile_Crossing"
    ] = any_crossing

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
        "Quantile_Version"
    ] = QUANTILE_VERSION

    return (
        result,
        model_records,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_predictions(
    predictions: pd.DataFrame,
) -> None:

    expected_fold_names = {
        fold["Fold_Name"]
        for fold in FOLDS
    }

    observed_fold_names = set(
        predictions[
            "Fold_Name"
        ]
        .unique()
        .tolist()
    )

    if observed_fold_names != expected_fold_names:

        raise ValueError(
            "Unexpected quantile fold set. "
            f"Expected {sorted(expected_fold_names)}; "
            f"found {sorted(observed_fold_names)}."
        )

    duplicate_count = int(
        predictions.duplicated(
            subset=[
                "Fold_Name",
                "Medicine_Key",
                "Target_Year",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            "Duplicate OOS quantile "
            f"records detected: {duplicate_count}"
        )

    for fold in FOLDS:

        fold_name = fold[
            "Fold_Name"
        ]

        test_year = fold[
            "Test_Year"
        ]

        fold_df = predictions.loc[
            predictions[
                "Fold_Name"
            ].eq(
                fold_name
            )
        ]

        if fold_df.empty:

            raise ValueError(
                f"{fold_name}: no quantile "
                "prediction rows."
            )

        observed_years = set(
            fold_df[
                "Target_Year"
            ]
            .unique()
            .tolist()
        )

        if observed_years != {
            test_year
        }:

            raise ValueError(
                f"{fold_name}: unexpected "
                f"target-year evidence "
                f"{observed_years}."
            )

        if (
            fold_df[
                "Medicine_Key"
            ].nunique()
            != len(fold_df)
        ):

            raise ValueError(
                f"{fold_name}: medicine-key "
                "structure failed."
            )

    for column in [
        "P50",
        "P80",
        "P90",
    ]:

        if predictions[
            column
        ].isna().any():

            raise ValueError(
                f"Missing {column} "
                f"forecast detected."
            )

        if (
            predictions[
                column
            ]
            .lt(0)
            .any()
        ):

            raise ValueError(
                f"Negative {column} "
                f"forecast detected."
            )

    if (
        predictions[
            "P50"
        ]
        .gt(
            predictions[
                "P80"
            ]
        )
        .any()
    ):

        raise ValueError(
            "Adjusted P50 > P80 detected."
        )

    if (
        predictions[
            "P80"
        ]
        .gt(
            predictions[
                "P90"
            ]
        )
        .any()
    ):

        raise ValueError(
            "Adjusted P80 > P90 detected."
        )

    duplicate_count = int(
        predictions.duplicated(
            subset=[
                "Fold_Name",
                "Medicine_Key",
                "Target_Year",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            "Duplicate quantile OOS "
            "prediction detected."
        )


# ============================================================
# QUANTILE METRICS
# ============================================================

def calculate_quantile_metric(
    dataframe: pd.DataFrame,
    label: str,
    quantile: float,
) -> dict:

    actual = (
        dataframe[
            "Target_Annual_Issuance"
        ]
        .to_numpy(
            dtype=float
        )
    )

    forecast = (
        dataframe[
            label
        ]
        .to_numpy(
            dtype=float
        )
    )

    coverage = float(
        np.mean(
            actual
            <= forecast
        )
    )

    positive_mask = (
        actual > 0
    )

    if positive_mask.any():

        positive_coverage = float(
            np.mean(
                actual[
                    positive_mask
                ]
                <=
                forecast[
                    positive_mask
                ]
            )
        )

        positive_pinball = float(
            mean_pinball_loss(
                actual[
                    positive_mask
                ],
                forecast[
                    positive_mask
                ],
                alpha=quantile,
            )
        )

    else:

        positive_coverage = np.nan
        positive_pinball = np.nan

    pinball = float(
        mean_pinball_loss(
            actual,
            forecast,
            alpha=quantile,
        )
    )

    return {
        "Rows":
            len(dataframe),

        "Positive_Targets":
            int(
                positive_mask.sum()
            ),

        "Nominal_Quantile":
            quantile,

        "Empirical_Coverage":
            coverage,

        "Empirical_Coverage_Pct":
            coverage
            * 100,

        "Coverage_Error_Pct_Points":
            (
                coverage
                -
                quantile
            )
            * 100,

        "Mean_Pinball_Loss":
            pinball,

        "Positive_Target_Coverage_Pct":
            (
                positive_coverage
                * 100
                if pd.notna(
                    positive_coverage
                )
                else np.nan
            ),

        "Positive_Target_Pinball_Loss":
            positive_pinball,
    }


# ============================================================
# OVERALL METRICS
# ============================================================

def build_overall_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    combined = (
        predictions.copy()
    )

    combined[
        "Fold_Name"
    ] = (
        COMBINED_OOS_LABEL
    )

    evaluation = pd.concat(
        [
            predictions,
            combined,
        ],
        ignore_index=True,
    )

    rows = []

    for fold_name, fold_df in (
        evaluation.groupby(
            "Fold_Name",
            sort=True,
        )
    ):

        for label, quantile in (
            QUANTILES.items()
        ):

            metrics = (
                calculate_quantile_metric(
                    fold_df,
                    label,
                    quantile,
                )
            )

            rows.append(
                {
                    "Fold_Name":
                        fold_name,

                    "Quantile_Label":
                        label,

                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SEGMENT METRICS
# ============================================================

def build_segment_metrics(
    predictions: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:

    combined = (
        predictions.copy()
    )

    combined[
        "Fold_Name"
    ] = (
        COMBINED_OOS_LABEL
    )

    evaluation = pd.concat(
        [
            predictions,
            combined,
        ],
        ignore_index=True,
    )

    rows = []

    for (
        fold_name,
        segment,
    ), group in evaluation.groupby(
        [
            "Fold_Name",
            segment_column,
        ],
        dropna=False,
        sort=True,
    ):

        segment_value = (
            str(segment)
            if pd.notna(segment)
            else "MISSING"
        )

        for label, quantile in (
            QUANTILES.items()
        ):

            metrics = (
                calculate_quantile_metric(
                    group,
                    label,
                    quantile,
                )
            )

            rows.append(
                {
                    "Fold_Name":
                        fold_name,

                    "Segment_Type":
                        segment_column,

                    "Segment":
                        segment_value,

                    "Quantile_Label":
                        label,

                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CROSSING SUMMARY
# ============================================================

def build_crossing_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for fold_name, fold_df in (
        predictions.groupby(
            "Fold_Name",
            sort=True,
        )
    ):

        rows.append(
            {
                "Fold_Name":
                    fold_name,

                "Rows":
                    len(
                        fold_df
                    ),

                "Raw_P50_P80_Crossings":
                    int(
                        fold_df[
                            "Raw_P50_P80_Crossing"
                        ]
                        .sum()
                    ),

                "Raw_P80_P90_Crossings":
                    int(
                        fold_df[
                            "Raw_P80_P90_Crossing"
                        ]
                        .sum()
                    ),

                "Raw_Any_Crossing":
                    int(
                        fold_df[
                            "Raw_Any_Quantile_Crossing"
                        ]
                        .sum()
                    ),

                "Rows_Adjusted":
                    int(
                        fold_df[
                            "Quantile_Adjustment_Applied"
                        ]
                        .sum()
                    ),

                "Raw_Any_Crossing_Pct":
                    float(
                        fold_df[
                            "Raw_Any_Quantile_Crossing"
                        ]
                        .mean()
                        * 100
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MODEL GOVERNANCE RECORD
# ============================================================

def build_governance_record() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "Decision_Version":
                    "POINT_MODEL_DECISION_V1",

                "Point_Forecast_Incumbent":
                    "NAIVE_LAG1",

                "Point_Forecast_Challenger":
                    "HURDLE_HGB_EXPECTED",

                "Occurrence_Risk_Model":
                    "HURDLE_HGB_OCCURRENCE",

                "Uncertainty_Model_Status":
                    "UNDER_BACKTEST",

                "Reason_WAPE":
                    (
                        "Hurdle observed combined "
                        "WAPE lower, but paired "
                        "95% bootstrap difference "
                        "interval includes zero."
                    ),

                "Reason_RMSE":
                    (
                        "Hurdle has lower RMSE "
                        "with paired difference "
                        "interval favoring Hurdle."
                    ),

                "Reason_Bias":
                    (
                        "Naive has substantially "
                        "lower absolute aggregate "
                        "forecast bias."
                    ),

                "Hybrid_Policy_Promoted":
                    False,

                "Production_2026_Forecast_Created":
                    False,
            }
        ]
    )


# ============================================================
# CONFIG
# ============================================================

def build_config(
    model_records: list[dict],
) -> pd.DataFrame:

    records = []

    for record in (
        model_records
    ):

        records.append(
            {
                **record,

                "Loss":
                    "quantile",

                "Learning_Rate":
                    0.05,

                "Max_Iter":
                    300,

                "Max_Leaf_Nodes":
                    15,

                "Min_Samples_Leaf":
                    20,

                "L2_Regularization":
                    1.0,

                "Early_Stopping":
                    False,

                "Random_State":
                    RANDOM_STATE,

                "Holdout_Tuning":
                    False,

                "Postprocess":
                    (
                        "clip_zero_then_"
                        "cumulative_max_for_"
                        "P50_P80_P90_monotonicity"
                    ),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# WRITE DATABASE
# ============================================================

def write_database(
    predictions: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    abc_metrics: pd.DataFrame,
    behavior_metrics: pd.DataFrame,
    crossing_summary: pd.DataFrame,
    config: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "predictions_df",
            predictions,
        )

        connection.register(
            "overall_df",
            overall_metrics,
        )

        connection.register(
            "abc_df",
            abc_metrics,
        )

        connection.register(
            "behavior_df",
            behavior_metrics,
        )

        connection.register(
            "crossing_df",
            crossing_summary,
        )

        connection.register(
            "config_df",
            config,
        )

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_predictions AS
            SELECT *
            FROM predictions_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_metrics_overall AS
            SELECT *
            FROM overall_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_metrics_by_abc AS
            SELECT *
            FROM abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_metrics_by_behavior AS
            SELECT *
            FROM behavior_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_crossing_summary AS
            SELECT *
            FROM crossing_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_model_config AS
            SELECT *
            FROM config_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_model_governance AS
            SELECT *
            FROM governance_df
            """
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 108)
    print("CDH INVENTORY DSS")
    print("CHRONOLOGICAL P50 / P80 / P90 QUANTILE BACKTEST")
    print("=" * 108)
    print()

    supervised = (
        load_supervised()
    )

    prediction_frames = []
    all_model_records = []

    for fold in FOLDS:

        predictions, records = (
            run_fold(
                supervised,
                fold,
            )
        )

        prediction_frames.append(
            predictions
        )

        all_model_records.extend(
            records
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    validate_predictions(
        predictions
    )

    print()
    print(
        "Quantile structural validation: PASS"
    )

    overall_metrics = (
        build_overall_metrics(
            predictions
        )
    )

    abc_metrics = (
        build_segment_metrics(
            predictions,
            "ABC_AsOf",
        )
    )

    behavior_metrics = (
        build_segment_metrics(
            predictions,
            "Behavior_Segment_AsOf",
        )
    )

    crossing_summary = (
        build_crossing_summary(
            predictions
        )
    )

    config = (
        build_config(
            all_model_records
        )
    )

    governance = (
        build_governance_record()
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    predictions.to_parquet(
        OUTPUT_DIR
        / "forecast_quantile_predictions.parquet",
        index=False,
        engine="pyarrow",
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "forecast_quantile_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_metrics.to_csv(
        OUTPUT_DIR
        / "forecast_quantile_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    abc_metrics.to_csv(
        OUTPUT_DIR
        / "forecast_quantile_metrics_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    behavior_metrics.to_csv(
        OUTPUT_DIR
        / "forecast_quantile_metrics_by_behavior.csv",
        index=False,
        encoding="utf-8-sig",
    )

    crossing_summary.to_csv(
        OUTPUT_DIR
        / "forecast_quantile_crossing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config.to_csv(
        OUTPUT_DIR
        / "forecast_quantile_model_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "forecast_model_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        predictions=predictions,
        overall_metrics=overall_metrics,
        abc_metrics=abc_metrics,
        behavior_metrics=behavior_metrics,
        crossing_summary=crossing_summary,
        config=config,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 108)
    print("OVERALL QUANTILE CALIBRATION")
    print("-" * 108)

    print(
        overall_metrics.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("QUANTILE CROSSING DIAGNOSTIC")
    print("-" * 108)

    print(
        crossing_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("COMBINED CALIBRATION BY AS-OF BEHAVIOR")
    print("-" * 108)

    combined_behavior = (
        behavior_metrics.loc[
            behavior_metrics[
                "Fold_Name"
            ]
            .eq(
                COMBINED_OOS_LABEL
            )
        ]
        .sort_values(
            [
                "Segment",
                "Nominal_Quantile",
            ]
        )
    )

    print(
        combined_behavior[
            [
                "Segment",
                "Quantile_Label",
                "Rows",
                "Positive_Targets",
                "Nominal_Quantile",
                "Empirical_Coverage_Pct",
                "Coverage_Error_Pct_Points",
                "Mean_Pinball_Loss",
                "Positive_Target_Coverage_Pct",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("MODEL GOVERNANCE")
    print("-" * 108)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  Empirical coverage is the "
        "percentage of observed annual "
        "issuance values that were less "
        "than or equal to the predicted "
        "quantile."
    )

    print(
        "  Pinball loss evaluates quantile "
        "forecast accuracy; lower is better."
    )

    print(
        "  Quantile models were trained "
        "chronologically with no random "
        "holdout split and no holdout-driven "
        "hyperparameter tuning."
    )

    print(
        "  Independently estimated quantiles "
        "may cross; raw crossing frequency "
        "is reported and operational outputs "
        "are monotonically adjusted."
    )

    print(
        "  P50/P80/P90 will NOT be described "
        "as exact service probabilities unless "
        "their empirical calibration supports "
        "that interpretation."
    )

    print()
    print("=" * 108)
    print(
        "QUANTILE BACKTEST COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()