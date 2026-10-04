from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "cdh_inventory.duckdb"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecast_2026_candidate_v1"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
    / "forecast_2026_candidate_v1"
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

FORECAST_VERSION = "FORECAST_2026_CANDIDATE_V1"

TARGET_YEAR = 2026
FEATURE_CUTOFF_YEAR = 2025

POINT_FORECAST_INCUMBENT = "NAIVE_LAG1"
POINT_FORECAST_CHALLENGER = "HURDLE_HGB_EXPECTED"

RANDOM_STATE = 42

QUANTILES = {
    "P50": 0.50,
    "P80": 0.80,
    "P90": 0.90,
}


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

def load_data() -> tuple[
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
            """
            SELECT *
            FROM forecasting_scoring_2026
            ORDER BY Medicine_Key
            """
        ).df()

    finally:

        connection.close()

    return (
        supervised,
        scoring,
    )


def load_quantile_validation() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        overall = connection.execute(
            """
            SELECT *
            FROM forecast_quantile_metrics_overall
            WHERE Fold_Name = 'COMBINED_OOS_2024_2025'
            ORDER BY Nominal_Quantile
            """
        ).df()

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
        overall,
        calibration,
    )


# ============================================================
# STRUCTURAL INPUT VALIDATION
# ============================================================

def validate_inputs(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
) -> None:

    if len(supervised) != 10932:

        raise ValueError(
            f"Expected 10,932 supervised "
            f"rows; found {len(supervised):,}"
        )

    if len(scoring) != 2733:

        raise ValueError(
            f"Expected 2,733 2026 scoring "
            f"rows; found {len(scoring):,}"
        )

    if set(
        supervised[
            "Target_Year"
        ]
        .unique()
        .tolist()
    ) != {
        2022,
        2023,
        2024,
        2025,
    }:

        raise ValueError(
            "Unexpected supervised target years."
        )

    if not (
        scoring[
            "Target_Year"
        ]
        .eq(
            TARGET_YEAR
        )
        .all()
    ):

        raise ValueError(
            "Scoring target year is not 2026."
        )

    if not (
        scoring[
            "Feature_Cutoff_Year"
        ]
        .eq(
            FEATURE_CUTOFF_YEAR
        )
        .all()
    ):

        raise ValueError(
            "2026 scoring cutoff is not 2025."
        )

    if not (
        scoring[
            "Max_History_Year_Used"
        ]
        .eq(
            FEATURE_CUTOFF_YEAR
        )
        .all()
    ):

        raise ValueError(
            "2026 scoring contains an "
            "unexpected history year."
        )

    if (
        scoring[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate scoring Medicine_Key."
        )


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
# PREPROCESSOR
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
# HURDLE MODELS
# ============================================================

def make_occurrence_pipeline() -> Pipeline:

    model = (
        HistGradientBoostingClassifier(
            loss="log_loss",
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
                model,
            ),
        ]
    )


def make_positive_amount_pipeline() -> Pipeline:

    model = (
        HistGradientBoostingRegressor(
            loss="squared_error",
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
                model,
            ),
        ]
    )


# ============================================================
# QUANTILE MODEL
# ============================================================

def make_quantile_pipeline(
    quantile: float,
) -> Pipeline:

    model = (
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
                model,
            ),
        ]
    )


# ============================================================
# TRAIN HURDLE
# ============================================================

def train_hurdle(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Pipeline,
    Pipeline,
]:

    print(
        "Training final hurdle occurrence "
        "classifier..."
    )

    x_train = prepare_x(
        supervised
    )

    x_score = prepare_x(
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
        occurrence_model
        .predict_proba(
            x_score
        )[:, 1]
    )

    probability = np.clip(
        probability,
        0,
        1,
    )

    print(
        "Training final hurdle "
        "positive-quantity regressor..."
    )

    positive_train = (
        supervised.loc[
            supervised[
                "Target_Annual_Issuance"
            ]
            .gt(0)
        ]
        .copy()
    )

    positive_x = prepare_x(
        positive_train
    )

    positive_y_log = np.log1p(
        positive_train[
            "Target_Annual_Issuance"
        ]
        .to_numpy(
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

    challenger_forecast = (
        probability
        *
        conditional_quantity
    )

    return (
        probability,
        conditional_quantity,
        challenger_forecast,
        occurrence_model,
        amount_model,
    )


# ============================================================
# TRAIN QUANTILES
# ============================================================

def train_quantiles(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
) -> tuple[
    dict,
    dict,
]:

    x_train = prepare_x(
        supervised
    )

    x_score = prepare_x(
        scoring
    )

    y_train = (
        supervised[
            "Target_Annual_Issuance"
        ]
        .to_numpy(
            dtype=float
        )
    )

    predictions = {}
    models = {}

    for label, quantile in (
        QUANTILES.items()
    ):

        print(
            f"Training final quantile "
            f"{label} ({quantile:.2f})..."
        )

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


# ============================================================
# BUILD FORECAST PACKAGE
# ============================================================

def build_package(
    scoring: pd.DataFrame,
    occurrence_probability: np.ndarray,
    conditional_positive: np.ndarray,
    hurdle_forecast: np.ndarray,
    quantile_predictions: dict,
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

    # --------------------------------------------------------
    # OFFICIAL INCUMBENT POINT FORECAST
    # --------------------------------------------------------

    result[
        "Point_Forecast_2026"
    ] = np.clip(
        result[
            "Lag1_Issuance"
        ]
        .to_numpy(
            dtype=float
        ),
        0,
        None,
    )

    result[
        "Point_Forecast_Model"
    ] = POINT_FORECAST_INCUMBENT

    result[
        "Point_Forecast_Status"
    ] = "INCUMBENT"

    # --------------------------------------------------------
    # HURDLE CHALLENGER
    # --------------------------------------------------------

    result[
        "Hurdle_Occurrence_Probability"
    ] = occurrence_probability

    result[
        "Hurdle_Conditional_Positive_Quantity"
    ] = conditional_positive

    result[
        "Hurdle_Point_Forecast_Challenger"
    ] = hurdle_forecast

    result[
        "Hurdle_Model_Status"
    ] = (
        "CHALLENGER_AND_OCCURRENCE_RISK"
    )

    # --------------------------------------------------------
    # RAW QUANTILES
    # --------------------------------------------------------

    raw_matrix = np.column_stack(
        [
            quantile_predictions[
                "P50"
            ],
            quantile_predictions[
                "P80"
            ],
            quantile_predictions[
                "P90"
            ],
        ]
    )

    raw_p50_p80_crossing = (
        raw_matrix[:, 0]
        >
        raw_matrix[:, 1]
    )

    raw_p80_p90_crossing = (
        raw_matrix[:, 1]
        >
        raw_matrix[:, 2]
    )

    raw_any_crossing = (
        raw_p50_p80_crossing
        |
        raw_p80_p90_crossing
    )

    # Same operational monotonicity rule
    # used in backtesting.
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
    ] = raw_any_crossing

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
        result[
            "P80"
        ]
        -
        result[
            "P50"
        ]
    )

    result[
        "P90_minus_P50"
    ] = (
        result[
            "P90"
        ]
        -
        result[
            "P50"
        ]
    )

    # --------------------------------------------------------
    # UNCERTAINTY GOVERNANCE
    # --------------------------------------------------------

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

    result[
        "Residual_Calibration_Applied"
    ] = False

    result[
        "Uncertainty_Policy"
    ] = (
        "RAW_BACKTESTED_QUANTILES_NO_"
        "TEMPORAL_RESIDUAL_RECALIBRATION"
    )

    result[
        "Forecast_Version"
    ] = FORECAST_VERSION

    result[
        "Observed_2026_Issuance_Used"
    ] = False

    return result


# ============================================================
# VALIDATE FINAL PACKAGE
# ============================================================

def validate_package(
    result: pd.DataFrame,
) -> None:

    if len(
        result
    ) != 2733:

        raise ValueError(
            f"Expected 2,733 forecast rows; "
            f"found {len(result):,}"
        )

    if (
        result[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate Medicine_Key "
            "in 2026 package."
        )

    required_numeric = [
        "Point_Forecast_2026",
        "Hurdle_Occurrence_Probability",
        "Hurdle_Point_Forecast_Challenger",
        "P50",
        "P80",
        "P90",
    ]

    for column in required_numeric:

        if result[
            column
        ].isna().any():

            raise ValueError(
                f"Missing {column}."
            )

    quantity_columns = [
        "Point_Forecast_2026",
        "Hurdle_Point_Forecast_Challenger",
        "P50",
        "P80",
        "P90",
    ]

    if (
        result[
            quantity_columns
        ]
        .lt(0)
        .any()
        .any()
    ):

        raise ValueError(
            "Negative forecast quantity."
        )

    if (
        result[
            "Hurdle_Occurrence_Probability"
        ]
        .lt(0)
        .any()
        or
        result[
            "Hurdle_Occurrence_Probability"
        ]
        .gt(1)
        .any()
    ):

        raise ValueError(
            "Occurrence probability "
            "outside [0,1]."
        )

    if (
        result[
            "P50"
        ]
        .gt(
            result[
                "P80"
            ]
        )
        .any()
    ):

        raise ValueError(
            "P50 > P80."
        )

    if (
        result[
            "P80"
        ]
        .gt(
            result[
                "P90"
            ]
        )
        .any()
    ):

        raise ValueError(
            "P80 > P90."
        )

    if not (
        result[
            "Target_Year"
        ]
        .eq(2026)
        .all()
    ):

        raise ValueError(
            "Unexpected target year."
        )

    if not (
        result[
            "Feature_Cutoff_Year"
        ]
        .eq(2025)
        .all()
    ):

        raise ValueError(
            "Unexpected feature cutoff year."
        )

    if (
        result[
            "Observed_2026_Issuance_Used"
        ]
        .any()
    ):

        raise ValueError(
            "2026 outcome leakage flag "
            "was unexpectedly TRUE."
        )


# ============================================================
# SEGMENT SUMMARY
# ============================================================

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

            segment_name = (
                str(segment)
                if pd.notna(segment)
                else "MISSING"
            )

            rows.append(
                {
                    "Segment_Type":
                        segment_type,

                    "Segment":
                        segment_name,

                    "Medicine_Count":
                        len(group),

                    "Point_Forecast_Total":
                        float(
                            group[
                                "Point_Forecast_2026"
                            ]
                            .sum()
                        ),

                    "Hurdle_Challenger_Total":
                        float(
                            group[
                                "Hurdle_Point_Forecast_Challenger"
                            ]
                            .sum()
                        ),

                    "P50_Total":
                        float(
                            group[
                                "P50"
                            ]
                            .sum()
                        ),

                    "P80_Total":
                        float(
                            group[
                                "P80"
                            ]
                            .sum()
                        ),

                    "P90_Total":
                        float(
                            group[
                                "P90"
                            ]
                            .sum()
                        ),

                    "Mean_Occurrence_Probability":
                        float(
                            group[
                                "Hurdle_Occurrence_Probability"
                            ]
                            .mean()
                        ),

                    "Median_Occurrence_Probability":
                        float(
                            group[
                                "Hurdle_Occurrence_Probability"
                            ]
                            .median()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TRAINING SUMMARY
# ============================================================

def build_training_summary(
    supervised: pd.DataFrame,
) -> pd.DataFrame:

    positive_rows = int(
        supervised[
            "Target_Annual_Issuance"
        ]
        .gt(0)
        .sum()
    )

    rows = [
        {
            "Component":
                "NAIVE_LAG1",

            "Role":
                "POINT_FORECAST_INCUMBENT",

            "Training_Rows":
                0,

            "Training_Target_Years":
                "N/A",

            "Target":
                "Previous completed-year issuance",

            "2026_Outcome_Used":
                False,
        },
        {
            "Component":
                "HURDLE_OCCURRENCE",

            "Role":
                "OCCURRENCE_RISK_CHALLENGER",

            "Training_Rows":
                len(
                    supervised
                ),

            "Training_Target_Years":
                "2022-2025",

            "Target":
                "Target_Occurrence",

            "2026_Outcome_Used":
                False,
        },
        {
            "Component":
                "HURDLE_POSITIVE_AMOUNT",

            "Role":
                "CHALLENGER_QUANTITY_COMPONENT",

            "Training_Rows":
                positive_rows,

            "Training_Target_Years":
                "2022-2025",

            "Target":
                "log1p positive annual issuance",

            "2026_Outcome_Used":
                False,
        },
    ]

    for label, quantile in (
        QUANTILES.items()
    ):

        rows.append(
            {
                "Component":
                    f"QUANTILE_{label}",

                "Role":
                    (
                        "UNCERTAINTY_MODEL"
                    ),

                "Training_Rows":
                    len(
                        supervised
                    ),

                "Training_Target_Years":
                    "2022-2025",

                "Target":
                    (
                        f"Raw annual issuance "
                        f"quantile {quantile:.2f}"
                    ),

                "2026_Outcome_Used":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance(
    quantile_validation: pd.DataFrame,
    calibration_validation: pd.DataFrame,
    result: pd.DataFrame,
) -> pd.DataFrame:

    coverage = {}

    for _, row in (
        quantile_validation.iterrows()
    ):

        coverage[
            row[
                "Quantile_Label"
            ]
        ] = float(
            row[
                "Empirical_Coverage_Pct"
            ]
        )

    calibration_flags = {}

    for _, row in (
        calibration_validation.iterrows()
    ):

        calibration_flags[
            row[
                "Quantile_Label"
            ]
        ] = {
            "coverage_improved":
                bool(
                    row[
                        "Calibration_Improved_Coverage_Error"
                    ]
                ),

            "pinball_improved":
                bool(
                    row[
                        "Calibration_Improved_Pinball"
                    ]
                ),
        }

    governance = pd.DataFrame(
        [
            {
                "Forecast_Version":
                    FORECAST_VERSION,

                "Target_Year":
                    2026,

                "Feature_Cutoff_Year":
                    2025,

                "Point_Forecast_Incumbent":
                    POINT_FORECAST_INCUMBENT,

                "Point_Forecast_Challenger":
                    POINT_FORECAST_CHALLENGER,

                "Occurrence_Risk_Model":
                    "HURDLE_HGB_OCCURRENCE",

                "P50_Combined_OOS_Coverage_Pct":
                    coverage.get(
                        "P50",
                        np.nan,
                    ),

                "P80_Combined_OOS_Coverage_Pct":
                    coverage.get(
                        "P80",
                        np.nan,
                    ),

                "P90_Combined_OOS_Coverage_Pct":
                    coverage.get(
                        "P90",
                        np.nan,
                    ),

                "Residual_Calibration_Promoted":
                    False,

                "Calibration_Decision":
                    (
                        "NOT_PROMOTED_BECAUSE_"
                        "TEMPORAL_VALIDATION_DID_"
                        "NOT_CONSISTENTLY_IMPROVE_"
                        "COVERAGE_AND_PINBALL"
                    ),

                "P50_Operational_Status":
                    "STATISTICAL_OUTPUT_ONLY",

                "P80_Operational_Status":
                    (
                        "PLANNING_SCENARIO_"
                        "NO_SERVICE_GUARANTEE"
                    ),

                "P90_Operational_Status":
                    (
                        "PLANNING_SCENARIO_"
                        "NO_SERVICE_GUARANTEE"
                    ),

                "Quantile_Crossing_Rows_2026":
                    int(
                        result[
                            "Raw_Quantile_Crossing"
                        ]
                        .sum()
                    ),

                "Quantile_Adjusted_Rows_2026":
                    int(
                        result[
                            "Quantile_Adjustment_Applied"
                        ]
                        .sum()
                    ),

                "Observed_2026_Issuance_Used":
                    False,

                "Production_Status":
                    "CANDIDATE_NOT_AUTOMATIC_PO",
            }
        ]
    )

    return governance


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(
    occurrence_model: Pipeline,
    amount_model: Pipeline,
    quantile_models: dict,
) -> None:

    joblib.dump(
        {
            "pipeline":
                occurrence_model,

            "role":
                "HURDLE_OCCURRENCE",

            "training_target_years":
                "2022-2025",

            "feature_cutoff_for_scoring":
                2025,

            "target_year":
                2026,

            "forecast_version":
                FORECAST_VERSION,
        },
        MODEL_DIR
        / "hurdle_occurrence_2026.joblib",
    )

    joblib.dump(
        {
            "pipeline":
                amount_model,

            "role":
                "HURDLE_POSITIVE_AMOUNT",

            "target_transform":
                "log1p_positive_only",

            "training_target_years":
                "2022-2025",

            "feature_cutoff_for_scoring":
                2025,

            "target_year":
                2026,

            "forecast_version":
                FORECAST_VERSION,
        },
        MODEL_DIR
        / "hurdle_positive_amount_2026.joblib",
    )

    for label, model in (
        quantile_models.items()
    ):

        joblib.dump(
            {
                "pipeline":
                    model,

                "role":
                    "QUANTILE",

                "quantile_label":
                    label,

                "quantile":
                    QUANTILES[
                        label
                    ],

                "training_target_years":
                    "2022-2025",

                "feature_cutoff_for_scoring":
                    2025,

                "target_year":
                    2026,

                "residual_calibration":
                    "NOT_APPLIED",

                "forecast_version":
                    FORECAST_VERSION,
            },
            MODEL_DIR
            / (
                f"quantile_"
                f"{label.lower()}_2026.joblib"
            ),
        )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    result: pd.DataFrame,
    segment_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:

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
            "governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_2026_candidate AS
            SELECT *
            FROM forecast_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_2026_segment_summary AS
            SELECT *
            FROM segment_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_2026_training_summary AS
            SELECT *
            FROM training_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_2026_governance AS
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
    print("2026 CANDIDATE ANNUAL FORECAST PACKAGE")
    print("=" * 108)
    print()

    supervised, scoring = (
        load_data()
    )

    (
        quantile_validation,
        calibration_validation,
    ) = load_quantile_validation()

    validate_inputs(
        supervised,
        scoring,
    )

    print(
        "Forecast input validation: PASS"
    )

    print(
        "Supervised training rows:",
        f"{len(supervised):,}",
    )

    print(
        "2026 scoring rows:",
        f"{len(scoring):,}",
    )

    print(
        "Feature cutoff year:",
        FEATURE_CUTOFF_YEAR,
    )

    print(
        "Observed 2026 issuance used: NO"
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
        quantile_predictions,
        quantile_models,
    ) = train_quantiles(
        supervised,
        scoring,
    )

    result = build_package(
        scoring=scoring,
        occurrence_probability=(
            occurrence_probability
        ),
        conditional_positive=(
            conditional_positive
        ),
        hurdle_forecast=(
            hurdle_forecast
        ),
        quantile_predictions=(
            quantile_predictions
        ),
    )

    validate_package(
        result
    )

    print()
    print(
        "2026 forecast package "
        "structural validation: PASS"
    )

    segment_summary = (
        build_segment_summary(
            result
        )
    )

    training_summary = (
        build_training_summary(
            supervised
        )
    )

    governance = (
        build_governance(
            quantile_validation=(
                quantile_validation
            ),
            calibration_validation=(
                calibration_validation
            ),
            result=result,
        )
    )

    save_models(
        occurrence_model=(
            occurrence_model
        ),
        amount_model=(
            amount_model
        ),
        quantile_models=(
            quantile_models
        ),
    )

    # --------------------------------------------------------
    # SAVE DATA OUTPUTS
    # --------------------------------------------------------

    result.to_parquet(
        OUTPUT_DIR
        / "forecast_2026_candidate.parquet",
        index=False,
        engine="pyarrow",
    )

    result.to_csv(
        OUTPUT_DIR
        / "forecast_2026_candidate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    segment_summary.to_csv(
        OUTPUT_DIR
        / "forecast_2026_segment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    training_summary.to_csv(
        OUTPUT_DIR
        / "forecast_2026_training_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "forecast_2026_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        result=result,
        segment_summary=segment_summary,
        training_summary=training_summary,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 108)
    print("2026 FORECAST TOTALS")
    print("-" * 108)

    print(
        "Incumbent point forecast "
        "(NAIVE_LAG1):",
        f"{result['Point_Forecast_2026'].sum():,.2f}",
    )

    print(
        "Hurdle challenger total:",
        f"{result['Hurdle_Point_Forecast_Challenger'].sum():,.2f}",
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

    print()
    print(
        "Raw quantile crossing rows:",
        int(
            result[
                "Raw_Quantile_Crossing"
            ]
            .sum()
        ),
    )

    print(
        "Rows receiving monotonic "
        "quantile adjustment:",
        int(
            result[
                "Quantile_Adjustment_Applied"
            ]
            .sum()
        ),
    )

    print()
    print("-" * 108)
    print("2026 BY AS-OF BEHAVIOR")
    print("-" * 108)

    behavior_summary = (
        segment_summary.loc[
            segment_summary[
                "Segment_Type"
            ]
            .eq(
                "Behavior_Segment_AsOf"
            )
        ]
    )

    print(
        behavior_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("MODEL TRAINING SUMMARY")
    print("-" * 108)

    print(
        training_summary.to_string(
            index=False
        )
    )

    print()
    print("-" * 108)
    print("2026 GOVERNANCE")
    print("-" * 108)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 108)
    print("IMPORTANT INTERPRETATION")
    print("-" * 108)

    print(
        "Point_Forecast_2026 is the "
        "governed incumbent forecast and "
        "uses 2025 recorded annual issuance."
    )

    print(
        "Hurdle occurrence probability is "
        "a model-based activation-risk "
        "indicator; it is not a clinical "
        "probability."
    )

    print(
        "The hurdle quantity forecast remains "
        "a challenger because its WAPE/MAE "
        "advantage over naive was not stable "
        "in paired bootstrap analysis."
    )

    print(
        "P50 is retained as a statistical "
        "quantile output and is NOT the "
        "production point forecast."
    )

    print(
        "P80 and P90 are model-based planning "
        "scenarios, not exact service-level "
        "guarantees."
    )

    print(
        "Temporal residual calibration was "
        "NOT applied because it did not "
        "consistently improve untouched 2025 "
        "coverage and pinball loss."
    )

    print(
        "No procurement order is created "
        "automatically by this forecast."
    )

    print()
    print("=" * 108)
    print(
        "2026 CANDIDATE FORECAST PACKAGE COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()