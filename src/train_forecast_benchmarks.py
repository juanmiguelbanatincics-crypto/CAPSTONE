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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

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
    / "benchmark_v1"
)

MODEL_DIR = (
    RUNTIME_CONTEXT.models_dir
    / "forecasting"
    / "benchmark_v1"
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

BENCHMARK_VERSION = "FORECAST_BENCHMARK_V1"
RANDOM_STATE = 42

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
# LOAD SUPERVISED DATA
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
# PREPROCESSORS
# ============================================================

def make_categorical_pipeline() -> Pipeline:

    return Pipeline(
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


def make_elastic_net_pipeline() -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                make_categorical_pipeline(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    estimator = ElasticNet(
        alpha=0.01,
        l1_ratio=0.50,
        max_iter=20000,
        tol=1e-4,
        selection="cyclic",
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def make_hist_gradient_boosting_pipeline() -> Pipeline:

    # Median imputation is performed only
    # inside each chronological training fold.
    # Missingness indicators retain information
    # that a source value was absent.

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                make_categorical_pipeline(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    estimator = (
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
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def build_model_config() -> pd.DataFrame:

    rows = [
        {
            "Model":
                "NAIVE_LAG1",

            "Model_Type":
                "BASELINE",

            "Target_Transformation":
                "NONE",

            "Configuration":
                json.dumps(
                    {
                        "forecast":
                            "previous completed year issuance"
                    }
                ),

            "Hyperparameter_Tuning":
                False,
        },
        {
            "Model":
                "MOVING_AVERAGE_2",

            "Model_Type":
                "BASELINE",

            "Target_Transformation":
                "NONE",

            "Configuration":
                json.dumps(
                    {
                        "forecast":
                            "mean of lag1 and lag2 issuance"
                    }
                ),

            "Hyperparameter_Tuning":
                False,
        },
        {
            "Model":
                "ELASTIC_NET_GLOBAL",

            "Model_Type":
                "GLOBAL_MACHINE_LEARNING",

            "Target_Transformation":
                "LOG1P",

            "Configuration":
                json.dumps(
                    {
                        "alpha": 0.01,
                        "l1_ratio": 0.50,
                        "max_iter": 20000,
                        "numeric_imputation":
                            "training-fold median + missing indicator",
                        "numeric_scaling":
                            "StandardScaler fit on training fold",
                        "categorical":
                            "training-fold one-hot encoding",
                    },
                    sort_keys=True,
                ),

            "Hyperparameter_Tuning":
                False,
        },
        {
            "Model":
                "HIST_GRADIENT_BOOSTING_GLOBAL",

            "Model_Type":
                "GLOBAL_MACHINE_LEARNING",

            "Target_Transformation":
                "LOG1P",

            "Configuration":
                json.dumps(
                    {
                        "loss":
                            "squared_error",
                        "learning_rate":
                            0.05,
                        "max_iter":
                            300,
                        "max_leaf_nodes":
                            15,
                        "min_samples_leaf":
                            20,
                        "l2_regularization":
                            1.0,
                        "early_stopping":
                            False,
                        "random_state":
                            RANDOM_STATE,
                        "numeric_imputation":
                            "training-fold median + missing indicator",
                        "categorical":
                            "training-fold one-hot encoding",
                    },
                    sort_keys=True,
                ),

            "Hyperparameter_Tuning":
                False,
        },
    ]

    config = pd.DataFrame(
        rows
    )

    config[
        "Benchmark_Version"
    ] = BENCHMARK_VERSION

    config[
        "Numeric_Features"
    ] = ", ".join(
        NUMERIC_FEATURES
    )

    config[
        "Categorical_Features"
    ] = ", ".join(
        CATEGORICAL_FEATURES
    )

    return config


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

def predict_naive(
    test: pd.DataFrame,
) -> np.ndarray:

    predictions = (
        pd.to_numeric(
            test[
                "Lag1_Issuance"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        predictions
    ).all():

        raise ValueError(
            "NAIVE_LAG1 contains "
            "missing/non-finite prediction."
        )

    return np.clip(
        predictions,
        0,
        None,
    )


def predict_moving_average_2(
    test: pd.DataFrame,
) -> np.ndarray:

    predictions = (
        pd.to_numeric(
            test[
                "Moving_Average_2"
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        predictions
    ).all():

        raise ValueError(
            "MOVING_AVERAGE_2 contains "
            "missing/non-finite prediction "
            "in a holdout year."
        )

    return np.clip(
        predictions,
        0,
        None,
    )


# ============================================================
# MACHINE-LEARNING FIT / PREDICT
# ============================================================

def fit_predict_ml(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold_name: str,
) -> tuple[
    np.ndarray,
    Pipeline,
]:

    if (
        model_name
        == "ELASTIC_NET_GLOBAL"
    ):
        pipeline = (
            make_elastic_net_pipeline()
        )

    elif (
        model_name
        == "HIST_GRADIENT_BOOSTING_GLOBAL"
    ):
        pipeline = (
            make_hist_gradient_boosting_pipeline()
        )

    else:
        raise ValueError(
            f"Unknown ML model: "
            f"{model_name}"
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

    # --------------------------------------------------------
    # LOG1P TARGET
    # --------------------------------------------------------

    y_train_transformed = (
        np.log1p(
            y_train
        )
    )

    pipeline.fit(
        x_train,
        y_train_transformed,
    )

    predicted_log = (
        pipeline.predict(
            x_test
        )
    )

    predictions = np.expm1(
        predicted_log
    )

    predictions = np.clip(
        predictions,
        0,
        None,
    )

    if not np.isfinite(
        predictions
    ).all():

        raise ValueError(
            f"{model_name} produced "
            f"non-finite predictions "
            f"in {fold_name}."
        )

    return (
        predictions,
        pipeline,
    )


# ============================================================
# PREDICTION RECORDS
# ============================================================

def make_prediction_frame(
    test: pd.DataFrame,
    predictions: np.ndarray,
    fold_name: str,
    model_name: str,
    train_rows: int,
    train_start: int,
    train_end: int,
) -> pd.DataFrame:

    output = test[
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
        ]
    ].copy()

    output[
        "Fold_Name"
    ] = fold_name

    output[
        "Model"
    ] = model_name

    output[
        "Prediction"
    ] = predictions

    output[
        "Error"
    ] = (
        output[
            "Prediction"
        ]
        -
        output[
            "Target_Annual_Issuance"
        ]
    )

    output[
        "Absolute_Error"
    ] = (
        output[
            "Error"
        ]
        .abs()
    )

    output[
        "Squared_Error"
    ] = (
        output[
            "Error"
        ]
        ** 2
    )

    output[
        "Train_Target_Year_Start"
    ] = train_start

    output[
        "Train_Target_Year_End"
    ] = train_end

    output[
        "Train_Row_Count"
    ] = train_rows

    output[
        "Benchmark_Version"
    ] = BENCHMARK_VERSION

    return output


# ============================================================
# ONE FOLD
# ============================================================

def run_fold(
    supervised: pd.DataFrame,
    fold: dict,
) -> pd.DataFrame:

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

    if (
        RUNTIME_CONTEXT.is_baseline
        and
        len(test) != 2733
    ):

        raise ValueError(
            f"{fold_name}: expected "
            f"2,733 baseline test rows; "
            f"found {len(test):,}"
        )

    if test[
        "Medicine_Key"
    ].duplicated().any():

        raise ValueError(
            f"{fold_name}: duplicate "
            "Medicine_Key in test population."
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

    print(
        f"  Test rows: "
        f"{len(test):,}"
    )

    prediction_frames = []

    # --------------------------------------------------------
    # NAIVE
    # --------------------------------------------------------

    print(
        "  Fitting/predicting: "
        "NAIVE_LAG1"
    )

    naive_prediction = (
        predict_naive(
            test
        )
    )

    prediction_frames.append(
        make_prediction_frame(
            test=test,
            predictions=naive_prediction,
            fold_name=fold_name,
            model_name="NAIVE_LAG1",
            train_rows=len(train),
            train_start=train_start,
            train_end=train_end,
        )
    )

    # --------------------------------------------------------
    # MOVING AVERAGE
    # --------------------------------------------------------

    print(
        "  Fitting/predicting: "
        "MOVING_AVERAGE_2"
    )

    ma_prediction = (
        predict_moving_average_2(
            test
        )
    )

    prediction_frames.append(
        make_prediction_frame(
            test=test,
            predictions=ma_prediction,
            fold_name=fold_name,
            model_name="MOVING_AVERAGE_2",
            train_rows=len(train),
            train_start=train_start,
            train_end=train_end,
        )
    )

    # --------------------------------------------------------
    # ELASTIC NET
    # --------------------------------------------------------

    print(
        "  Fitting/predicting: "
        "ELASTIC_NET_GLOBAL"
    )

    elastic_prediction, elastic_model = (
        fit_predict_ml(
            model_name=(
                "ELASTIC_NET_GLOBAL"
            ),
            train=train,
            test=test,
            fold_name=fold_name,
        )
    )

    prediction_frames.append(
        make_prediction_frame(
            test=test,
            predictions=elastic_prediction,
            fold_name=fold_name,
            model_name=(
                "ELASTIC_NET_GLOBAL"
            ),
            train_rows=len(train),
            train_start=train_start,
            train_end=train_end,
        )
    )

    joblib.dump(
        {
            "pipeline":
                elastic_model,

            "target_transform":
                "log1p",

            "inverse_transform":
                "expm1_then_clip_zero",

            "numeric_features":
                NUMERIC_FEATURES,

            "categorical_features":
                CATEGORICAL_FEATURES,

            "fold":
                fold,

            "benchmark_version":
                BENCHMARK_VERSION,
        },
        MODEL_DIR
        / (
            f"{fold_name.lower()}"
            f"__elastic_net_global.joblib"
        ),
    )

    # --------------------------------------------------------
    # HIST GRADIENT BOOSTING
    # --------------------------------------------------------

    print(
        "  Fitting/predicting: "
        "HIST_GRADIENT_BOOSTING_GLOBAL"
    )

    hgb_prediction, hgb_model = (
        fit_predict_ml(
            model_name=(
                "HIST_GRADIENT_BOOSTING_GLOBAL"
            ),
            train=train,
            test=test,
            fold_name=fold_name,
        )
    )

    prediction_frames.append(
        make_prediction_frame(
            test=test,
            predictions=hgb_prediction,
            fold_name=fold_name,
            model_name=(
                "HIST_GRADIENT_BOOSTING_GLOBAL"
            ),
            train_rows=len(train),
            train_start=train_start,
            train_end=train_end,
        )
    )

    joblib.dump(
        {
            "pipeline":
                hgb_model,

            "target_transform":
                "log1p",

            "inverse_transform":
                "expm1_then_clip_zero",

            "numeric_features":
                NUMERIC_FEATURES,

            "categorical_features":
                CATEGORICAL_FEATURES,

            "fold":
                fold,

            "benchmark_version":
                BENCHMARK_VERSION,
        },
        MODEL_DIR
        / (
            f"{fold_name.lower()}"
            f"__hist_gradient_boosting_global.joblib"
        ),
    )

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> dict:

    actual = (
        dataframe[
            "Target_Annual_Issuance"
        ]
        .to_numpy(
            dtype=float
        )
    )

    predicted = (
        dataframe[
            "Prediction"
        ]
        .to_numpy(
            dtype=float
        )
    )

    absolute_error = np.abs(
        actual - predicted
    )

    actual_total = float(
        actual.sum()
    )

    forecast_total = float(
        predicted.sum()
    )

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        root_mean_squared_error(
            actual,
            predicted,
        )
    )

    if actual_total > 0:

        wape_pct = float(
            absolute_error.sum()
            /
            actual_total
            *
            100
        )

        bias_pct = float(
            (
                forecast_total
                - actual_total
            )
            /
            actual_total
            *
            100
        )

    else:

        wape_pct = np.nan
        bias_pct = np.nan

    return {
        "Rows":
            len(dataframe),

        "Positive_Targets":
            int(
                (
                    actual > 0
                ).sum()
            ),

        "Actual_Total":
            actual_total,

        "Forecast_Total":
            forecast_total,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "WAPE_Pct":
            wape_pct,

        "Forecast_Bias_Pct":
            bias_pct,
    }


# ============================================================
# BUILD EVALUATION POPULATION
# ============================================================

def build_evaluation_population(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    # Original fold-specific OOS predictions.
    fold_specific = (
        predictions.copy()
    )

    # Combined means:
    # 2024 predictions come from BACKTEST_2024
    # and 2025 predictions come from BACKTEST_2025.
    #
    # No in-sample predictions are inserted.
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
            fold_specific,
            combined,
        ],
        ignore_index=True,
    )

    return evaluation


# ============================================================
# OVERALL METRICS
# ============================================================

def build_overall_metrics(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        fold_name,
        model_name,
    ), group in evaluation.groupby(
        [
            "Fold_Name",
            "Model",
        ],
        sort=True,
    ):

        row = {
            "Fold_Name":
                fold_name,

            "Model":
                model_name,
        }

        row.update(
            calculate_metrics(
                group
            )
        )

        rows.append(
            row
        )

    metrics = pd.DataFrame(
        rows
    )

    metrics[
        "WAPE_Rank"
    ] = (
        metrics.groupby(
            "Fold_Name"
        )[
            "WAPE_Pct"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype("Int64")
    )

    metrics[
        "MAE_Rank"
    ] = (
        metrics.groupby(
            "Fold_Name"
        )[
            "MAE"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype("Int64")
    )

    return metrics.sort_values(
        [
            "Fold_Name",
            "WAPE_Rank",
            "MAE_Rank",
            "Model",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# SEGMENT METRICS
# ============================================================

def build_segment_metrics(
    evaluation: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:

    rows = []

    for (
        fold_name,
        model_name,
        segment,
    ), group in evaluation.groupby(
        [
            "Fold_Name",
            "Model",
            segment_column,
        ],
        dropna=False,
        sort=True,
    ):

        row = {
            "Fold_Name":
                fold_name,

            "Model":
                model_name,

            "Segment":
                (
                    str(segment)
                    if pd.notna(segment)
                    else "MISSING"
                ),
        }

        row.update(
            calculate_metrics(
                group
            )
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "Segment_Type"
    ] = segment_column

    return result[
        [
            "Fold_Name",
            "Model",
            "Segment_Type",
            "Segment",
            "Rows",
            "Positive_Targets",
            "Actual_Total",
            "Forecast_Total",
            "MAE",
            "RMSE",
            "WAPE_Pct",
            "Forecast_Bias_Pct",
        ]
    ]


# ============================================================
# VALIDATION
# ============================================================

def validate_predictions(
    predictions: pd.DataFrame,
) -> None:

    expected_models = {
        "NAIVE_LAG1",
        "MOVING_AVERAGE_2",
        "ELASTIC_NET_GLOBAL",
        "HIST_GRADIENT_BOOSTING_GLOBAL",
    }

    actual_models = set(
        predictions[
            "Model"
        ]
        .unique()
        .tolist()
    )

    if actual_models != expected_models:

        raise ValueError(
            "Unexpected benchmark model set. "
            f"Expected {sorted(expected_models)}; "
            f"found {sorted(actual_models)}."
        )

    if predictions[
        "Prediction"
    ].isna().any():

        raise ValueError(
            "Missing forecast detected."
        )

    if (
        predictions[
            "Prediction"
        ]
        .lt(0)
        .any()
    ):

        raise ValueError(
            "Negative forecast detected."
        )

    duplicate_count = int(
        predictions.duplicated(
            subset=[
                "Fold_Name",
                "Model",
                "Medicine_Key",
                "Target_Year",
            ]
        ).sum()
    )

    if duplicate_count:

        raise ValueError(
            "Duplicate out-of-sample "
            "prediction record detected."
        )

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
            "Unexpected benchmark fold set. "
            f"Expected {sorted(expected_fold_names)}; "
            f"found {sorted(observed_fold_names)}."
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
                f"target years "
                f"{observed_years}."
            )

        reference_keys = None
        reference_actual = None

        for model_name in sorted(
            expected_models
        ):

            model_df = (
                fold_df.loc[
                    fold_df[
                        "Model"
                    ].eq(
                        model_name
                    )
                ]
                .sort_values(
                    "Medicine_Key"
                )
                .reset_index(
                    drop=True
                )
            )

            if model_df.empty:

                raise ValueError(
                    f"{fold_name} / "
                    f"{model_name}: "
                    "no prediction rows."
                )

            if (
                RUNTIME_CONTEXT.is_baseline
                and
                len(model_df) != 2733
            ):

                raise ValueError(
                    f"{fold_name} / "
                    f"{model_name}: "
                    "expected 2,733 baseline rows; "
                    f"found {len(model_df):,}."
                )

            model_keys = (
                model_df[
                    "Medicine_Key"
                ]
                .tolist()
            )

            model_actual = (
                model_df[
                    "Target_Annual_Issuance"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            if reference_keys is None:

                reference_keys = model_keys
                reference_actual = (
                    model_actual.copy()
                )

            else:

                if model_keys != reference_keys:

                    raise ValueError(
                        f"{fold_name}: medicine "
                        "population differs "
                        "between models."
                    )

                if not np.allclose(
                    model_actual,
                    reference_actual,
                    atol=0,
                    rtol=0,
                ):

                    raise ValueError(
                        f"{fold_name}: target "
                        "values differ "
                        "between models."
                    )


# ============================================================
# WRITE DATABASE
# ============================================================

def write_database(
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
    abc_metrics: pd.DataFrame,
    behavior_metrics: pd.DataFrame,
    model_config: pd.DataFrame,
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
            overall,
        )

        connection.register(
            "abc_metrics_df",
            abc_metrics,
        )

        connection.register(
            "behavior_metrics_df",
            behavior_metrics,
        )

        connection.register(
            "model_config_df",
            model_config,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_benchmark_predictions AS
            SELECT *
            FROM predictions_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_benchmark_metrics_overall AS
            SELECT *
            FROM overall_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_benchmark_metrics_by_abc AS
            SELECT *
            FROM abc_metrics_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_benchmark_metrics_by_behavior AS
            SELECT *
            FROM behavior_metrics_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_benchmark_model_config AS
            SELECT *
            FROM model_config_df
            """
        )

    finally:
        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 104)
    print("CDH INVENTORY DSS")
    print("CHRONOLOGICAL ANNUAL FORECAST BENCHMARKS — V1")
    print("=" * 104)
    print()

    supervised = (
        load_supervised()
    )

    combined_year_range = (
        f"{FOLDS[0]['Test_Year']}-"
        f"{FOLDS[-1]['Test_Year']}"
    )

    print(
        "Supervised medicine-year rows:",
        f"{len(supervised):,}",
    )

    print(
        "Target years:",
        sorted(
            supervised[
                "Target_Year"
            ]
            .unique()
            .tolist()
        ),
    )

    print()
    print(
        "Forecast predictors:",
        len(
            ALL_FEATURES
        ),
    )

    print(
        "No Medicine_Key, Brand_Name, "
        "Generic_Name, Target_Year, or "
        "target-year outcome is used "
        "as a model predictor."
    )

    prediction_frames = []

    for fold in FOLDS:

        prediction_frames.append(
            run_fold(
                supervised,
                fold,
            )
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
        "Prediction structural validation: PASS"
    )

    evaluation = (
        build_evaluation_population(
            predictions
        )
    )

    overall_metrics = (
        build_overall_metrics(
            evaluation
        )
    )

    abc_metrics = (
        build_segment_metrics(
            evaluation,
            "ABC_AsOf",
        )
    )

    behavior_metrics = (
        build_segment_metrics(
            evaluation,
            "Behavior_Segment_AsOf",
        )
    )

    model_config = (
        build_model_config()
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    predictions.to_parquet(
        OUTPUT_DIR
        / "forecast_benchmark_predictions.parquet",
        index=False,
        engine="pyarrow",
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "forecast_benchmark_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall_metrics.to_csv(
        OUTPUT_DIR
        / "forecast_benchmark_metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    abc_metrics.to_csv(
        OUTPUT_DIR
        / "forecast_benchmark_metrics_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    behavior_metrics.to_csv(
        OUTPUT_DIR
        / "forecast_benchmark_metrics_by_behavior.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_config.to_csv(
        OUTPUT_DIR
        / "forecast_benchmark_model_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        predictions=predictions,
        overall=overall_metrics,
        abc_metrics=abc_metrics,
        behavior_metrics=behavior_metrics,
        model_config=model_config,
    )

    # --------------------------------------------------------
    # CONSOLE RESULTS
    # --------------------------------------------------------

    print()
    print("-" * 104)
    print("OVERALL OUT-OF-SAMPLE PERFORMANCE")
    print("-" * 104)

    display_overall = (
        overall_metrics[
            [
                "Fold_Name",
                "Model",
                "Rows",
                "Positive_Targets",
                "Actual_Total",
                "Forecast_Total",
                "MAE",
                "RMSE",
                "WAPE_Pct",
                "Forecast_Bias_Pct",
                "WAPE_Rank",
            ]
        ]
    )

    print(
        display_overall.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 104)
    print(f"COMBINED {combined_year_range} OOS — BY PRIOR ABC")
    print("-" * 104)

    combined_abc = (
        abc_metrics.loc[
            abc_metrics[
                "Fold_Name"
            ]
            .eq(
                COMBINED_OOS_LABEL
            )
        ]
        .sort_values(
            [
                "Segment",
                "WAPE_Pct",
                "Model",
            ]
        )
    )

    print(
        combined_abc[
            [
                "Model",
                "Segment",
                "Rows",
                "Positive_Targets",
                "Actual_Total",
                "MAE",
                "RMSE",
                "WAPE_Pct",
                "Forecast_Bias_Pct",
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
    print("-" * 104)
    print(f"COMBINED {combined_year_range} OOS — BY AS-OF BEHAVIOR")
    print("-" * 104)

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
                "WAPE_Pct",
                "Model",
            ]
        )
    )

    print(
        combined_behavior[
            [
                "Model",
                "Segment",
                "Rows",
                "Positive_Targets",
                "Actual_Total",
                "MAE",
                "RMSE",
                "WAPE_Pct",
                "Forecast_Bias_Pct",
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
    print("-" * 104)
    print("MODEL GOVERNANCE")
    print("-" * 104)

    print(
        "Backtests are chronological only."
    )

    for fold in FOLDS:
        print(
            f"{fold['Fold_Name']} trains on target "
            f"years {fold['Train_Start']}-"
            f"{fold['Train_End']} and tests "
            f"{fold['Test_Year']}."
        )

    print(
        "Machine-learning preprocessing "
        "is fitted on training rows only."
    )

    print(
        "No holdout-driven hyperparameter "
        "tuning was performed."
    )

    print(
        "HistGradientBoosting internal "
        "early stopping is disabled."
    )

    print(
        "ML targets use log1p; forecasts "
        "are inverse-transformed and "
        "clipped to zero."
    )

    print(
        "No production model is promoted "
        "by this script."
    )

    print()
    print("=" * 104)
    print(
        "FORECAST BENCHMARKING COMPLETE"
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
