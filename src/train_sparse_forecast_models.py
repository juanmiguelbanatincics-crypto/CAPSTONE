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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
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
    / "sparse_models_v1"
)

MODEL_DIR = (
    RUNTIME_CONTEXT.models_dir
    / "forecasting"
    / "sparse_models_v1"
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

MODEL_VERSION = "SPARSE_FORECAST_MODELS_V1"
RANDOM_STATE = 42
OCCURRENCE_THRESHOLD = 0.50


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


def load_benchmark_predictions() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        dataframe = connection.execute(
            """
            SELECT *
            FROM forecast_benchmark_predictions
            ORDER BY Fold_Name, Model, Medicine_Key
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
# POISSON MODEL
# ============================================================

def make_poisson_pipeline() -> Pipeline:

    estimator = (
        HistGradientBoostingRegressor(
            loss="poisson",
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
# HURDLE MODELS
# ============================================================

def make_occurrence_pipeline() -> Pipeline:

    estimator = (
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
                estimator,
            ),
        ]
    )


def make_positive_amount_pipeline() -> Pipeline:

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
                make_preprocessor(),
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# PREDICTION FRAME
# ============================================================

def make_prediction_frame(
    test: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    fold_name: str,
    train_start: int,
    train_end: int,
    train_rows: int,
    occurrence_probability=None,
    conditional_positive_prediction=None,
) -> pd.DataFrame:

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
        "Model"
    ] = model_name

    result[
        "Prediction"
    ] = np.clip(
        predictions,
        0,
        None,
    )

    result[
        "Error"
    ] = (
        result[
            "Prediction"
        ]
        -
        result[
            "Target_Annual_Issuance"
        ]
    )

    result[
        "Absolute_Error"
    ] = (
        result[
            "Error"
        ]
        .abs()
    )

    if occurrence_probability is None:

        result[
            "Occurrence_Probability"
        ] = np.nan

    else:

        result[
            "Occurrence_Probability"
        ] = (
            occurrence_probability
        )

    if (
        conditional_positive_prediction
        is None
    ):

        result[
            "Conditional_Positive_Prediction"
        ] = np.nan

    else:

        result[
            "Conditional_Positive_Prediction"
        ] = (
            conditional_positive_prediction
        )

    result[
        "Train_Target_Year_Start"
    ] = train_start

    result[
        "Train_Target_Year_End"
    ] = train_end

    result[
        "Train_Row_Count"
    ] = train_rows

    result[
        "Model_Version"
    ] = MODEL_VERSION

    return result


# ============================================================
# RUN ONE FOLD
# ============================================================

def run_fold(
    supervised: pd.DataFrame,
    fold: dict,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
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

    # --------------------------------------------------------
    # POISSON
    # --------------------------------------------------------

    print(
        "  Fitting: "
        "HIST_GRADIENT_BOOSTING_POISSON"
    )

    poisson_model = (
        make_poisson_pipeline()
    )

    poisson_model.fit(
        x_train,
        y_train,
    )

    poisson_prediction = (
        poisson_model.predict(
            x_test
        )
    )

    poisson_prediction = np.clip(
        poisson_prediction,
        0,
        None,
    )

    poisson_frame = (
        make_prediction_frame(
            test=test,
            predictions=poisson_prediction,
            model_name=(
                "HIST_GRADIENT_BOOSTING_POISSON"
            ),
            fold_name=fold_name,
            train_start=train_start,
            train_end=train_end,
            train_rows=len(train),
        )
    )

    joblib.dump(
        {
            "pipeline":
                poisson_model,
            "target":
                "raw_nonnegative_annual_issuance",
            "fold":
                fold,
            "model_version":
                MODEL_VERSION,
        },
        MODEL_DIR
        / (
            f"{fold_name.lower()}"
            "__hist_gradient_boosting_poisson.joblib"
        ),
    )

    # --------------------------------------------------------
    # HURDLE OCCURRENCE
    # --------------------------------------------------------

    print(
        "  Fitting: "
        "HURDLE occurrence classifier"
    )

    occurrence_model = (
        make_occurrence_pipeline()
    )

    occurrence_target = (
        train[
            "Target_Occurrence"
        ]
        .astype(int)
        .to_numpy()
    )

    occurrence_model.fit(
        x_train,
        occurrence_target,
    )

    probability = (
        occurrence_model
        .predict_proba(
            x_test
        )[:, 1]
    )

    probability = np.clip(
        probability,
        0,
        1,
    )

    # --------------------------------------------------------
    # HURDLE POSITIVE AMOUNT
    # --------------------------------------------------------

    print(
        "  Fitting: "
        "HURDLE positive-demand regressor"
    )

    positive_train = (
        train.loc[
            train[
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

    positive_model = (
        make_positive_amount_pipeline()
    )

    positive_model.fit(
        positive_x,
        positive_y_log,
    )

    conditional_log = (
        positive_model.predict(
            x_test
        )
    )

    conditional_quantity = np.expm1(
        conditional_log
    )

    conditional_quantity = np.clip(
        conditional_quantity,
        0,
        None,
    )

    # Expected unconditional quantity
    hurdle_prediction = (
        probability
        *
        conditional_quantity
    )

    hurdle_frame = (
        make_prediction_frame(
            test=test,
            predictions=hurdle_prediction,
            model_name=(
                "HURDLE_HGB_EXPECTED"
            ),
            fold_name=fold_name,
            train_start=train_start,
            train_end=train_end,
            train_rows=len(train),
            occurrence_probability=probability,
            conditional_positive_prediction=(
                conditional_quantity
            ),
        )
    )

    joblib.dump(
        {
            "occurrence_pipeline":
                occurrence_model,
            "positive_amount_pipeline":
                positive_model,
            "amount_target_transform":
                "log1p_positive_targets_only",
            "forecast_equation":
                (
                    "P(Y>0) * "
                    "E_positive_quantity"
                ),
            "occurrence_threshold_for_metrics":
                OCCURRENCE_THRESHOLD,
            "fold":
                fold,
            "model_version":
                MODEL_VERSION,
        },
        MODEL_DIR
        / (
            f"{fold_name.lower()}"
            "__hurdle_hgb_expected.joblib"
        ),
    )

    predictions = pd.concat(
        [
            poisson_frame,
            hurdle_frame,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # OCCURRENCE DIAGNOSTICS
    # --------------------------------------------------------

    actual_occurrence = (
        test[
            "Target_Occurrence"
        ]
        .astype(int)
        .to_numpy()
    )

    classified = (
        probability
        >= OCCURRENCE_THRESHOLD
    ).astype(int)

    prevalence = float(
        actual_occurrence.mean()
    )

    average_precision = float(
        average_precision_score(
            actual_occurrence,
            probability,
        )
    )

    occurrence_row = pd.DataFrame(
        [
            {
                "Fold_Name":
                    fold_name,

                "Target_Year":
                    test_year,

                "Rows":
                    len(test),

                "Positive_Targets":
                    int(
                        actual_occurrence.sum()
                    ),

                "Prevalence":
                    prevalence,

                "Threshold":
                    OCCURRENCE_THRESHOLD,

                "Precision":
                    float(
                        precision_score(
                            actual_occurrence,
                            classified,
                            zero_division=0,
                        )
                    ),

                "Recall":
                    float(
                        recall_score(
                            actual_occurrence,
                            classified,
                            zero_division=0,
                        )
                    ),

                "F1":
                    float(
                        f1_score(
                            actual_occurrence,
                            classified,
                            zero_division=0,
                        )
                    ),

                "Average_Precision":
                    average_precision,

                "AP_Lift_vs_Prevalence":
                    (
                        average_precision
                        / prevalence
                        if prevalence > 0
                        else np.nan
                    ),

                "Model_Version":
                    MODEL_VERSION,
            }
        ]
    )

    return (
        predictions,
        occurrence_row,
    )


# ============================================================
# FORECAST METRICS
# ============================================================

def calculate_forecast_metrics(
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

    forecast = (
        dataframe[
            "Prediction"
        ]
        .to_numpy(
            dtype=float
        )
    )

    absolute_error = np.abs(
        actual - forecast
    )

    actual_total = float(
        actual.sum()
    )

    forecast_total = float(
        forecast.sum()
    )

    return {
        "Rows":
            len(dataframe),

        "Actual_Total":
            actual_total,

        "Forecast_Total":
            forecast_total,

        "MAE":
            float(
                mean_absolute_error(
                    actual,
                    forecast,
                )
            ),

        "RMSE":
            float(
                root_mean_squared_error(
                    actual,
                    forecast,
                )
            ),

        "WAPE_Pct":
            (
                float(
                    absolute_error.sum()
                    /
                    actual_total
                    * 100
                )
                if actual_total > 0
                else np.nan
            ),

        "Forecast_Bias_Pct":
            (
                float(
                    (
                        forecast_total
                        - actual_total
                    )
                    /
                    actual_total
                    * 100
                )
                if actual_total > 0
                else np.nan
            ),
    }


def build_metrics(
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
            calculate_forecast_metrics(
                group
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# OCCURRENCE BY BEHAVIOR
# ============================================================

def build_occurrence_by_behavior(
    hurdle_predictions: pd.DataFrame,
) -> pd.DataFrame:

    hurdle = (
        hurdle_predictions.loc[
            hurdle_predictions[
                "Model"
            ]
            .eq(
                "HURDLE_HGB_EXPECTED"
            )
        ]
        .copy()
    )

    combined = (
        hurdle.copy()
    )

    combined[
        "Fold_Name"
    ] = (
        COMBINED_OOS_LABEL
    )

    evaluation = pd.concat(
        [
            hurdle,
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
            "Behavior_Segment_AsOf",
        ],
        dropna=False,
        sort=True,
    ):

        actual = (
            group[
                "Target_Occurrence"
            ]
            .astype(int)
            .to_numpy()
        )

        probability = (
            group[
                "Occurrence_Probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        classified = (
            probability
            >= OCCURRENCE_THRESHOLD
        ).astype(int)

        prevalence = float(
            actual.mean()
        )

        if (
            actual.sum() == 0
        ):

            average_precision = np.nan

        else:

            average_precision = float(
                average_precision_score(
                    actual,
                    probability,
                )
            )

        rows.append(
            {
                "Fold_Name":
                    fold_name,

                "Behavior_Segment_AsOf":
                    (
                        str(segment)
                        if pd.notna(segment)
                        else "MISSING"
                    ),

                "Rows":
                    len(group),

                "Positive_Targets":
                    int(
                        actual.sum()
                    ),

                "Prevalence":
                    prevalence,

                "Precision":
                    float(
                        precision_score(
                            actual,
                            classified,
                            zero_division=0,
                        )
                    ),

                "Recall":
                    float(
                        recall_score(
                            actual,
                            classified,
                            zero_division=0,
                        )
                    ),

                "F1":
                    float(
                        f1_score(
                            actual,
                            classified,
                            zero_division=0,
                        )
                    ),

                "Average_Precision":
                    average_precision,

                "AP_Lift_vs_Prevalence":
                    (
                        average_precision
                        / prevalence
                        if (
                            prevalence > 0
                            and
                            pd.notna(
                                average_precision
                            )
                        )
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_predictions(
    predictions: pd.DataFrame,
) -> None:

    expected_models = {
        "HIST_GRADIENT_BOOSTING_POISSON",
        "HURDLE_HGB_EXPECTED",
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
            "Unexpected sparse-model set. "
            f"Expected {sorted(expected_models)}; "
            f"found {sorted(actual_models)}."
        )

    if predictions[
        "Prediction"
    ].isna().any():

        raise ValueError(
            "Missing prediction detected."
        )

    if (
        predictions[
            "Prediction"
        ]
        .lt(0)
        .any()
    ):

        raise ValueError(
            "Negative prediction detected."
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
            "Duplicate new OOS "
            "prediction detected."
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
            "Unexpected sparse-model fold set. "
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
                        "between sparse models."
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
                        "between sparse models."
                    )


# ============================================================
# ALL-MODEL COMPARISON
# ============================================================

def build_all_model_comparison(
    benchmark_predictions: pd.DataFrame,
    new_predictions: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "Fold_Name",
        "Model",
        "Medicine_Key",
        "Target_Year",
        "Target_Annual_Issuance",
        "Prediction",
    ]

    benchmark = (
        benchmark_predictions[
            columns
        ]
        .copy()
    )

    new = (
        new_predictions[
            columns
        ]
        .copy()
    )

    all_predictions = pd.concat(
        [
            benchmark,
            new,
        ],
        ignore_index=True,
    )

    combined = (
        all_predictions.copy()
    )

    combined[
        "Fold_Name"
    ] = (
        COMBINED_OOS_LABEL
    )

    evaluation = pd.concat(
        [
            all_predictions,
            combined,
        ],
        ignore_index=True,
    )

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
            calculate_forecast_metrics(
                group
            )
        )

        rows.append(
            row
        )

    comparison = pd.DataFrame(
        rows
    )

    comparison[
        "WAPE_Rank"
    ] = (
        comparison.groupby(
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

    return comparison.sort_values(
        [
            "Fold_Name",
            "WAPE_Rank",
            "Model",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# MODEL CONFIG
# ============================================================

def build_model_config() -> pd.DataFrame:

    rows = [
        {
            "Model":
                "HIST_GRADIENT_BOOSTING_POISSON",

            "Purpose":
                "Nonnegative count-oriented "
                "global regression",

            "Target":
                "Raw annual issuance",

            "Configuration":
                json.dumps(
                    {
                        "loss":
                            "poisson",
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
                    },
                    sort_keys=True,
                ),

            "Holdout_Tuning":
                False,
        },

        {
            "Model":
                "HURDLE_HGB_EXPECTED",

            "Purpose":
                "Zero-heavy occurrence plus "
                "positive-quantity modeling",

            "Target":
                (
                    "Occurrence probability × "
                    "conditional positive quantity"
                ),

            "Configuration":
                json.dumps(
                    {
                        "occurrence_model":
                            "HistGradientBoostingClassifier",
                        "positive_model":
                            "HistGradientBoostingRegressor",
                        "positive_target":
                            "log1p",
                        "occurrence_threshold_metrics":
                            OCCURRENCE_THRESHOLD,
                        "expected_forecast":
                            (
                                "probability * "
                                "conditional_quantity"
                            ),
                        "early_stopping":
                            False,
                        "random_state":
                            RANDOM_STATE,
                    },
                    sort_keys=True,
                ),

            "Holdout_Tuning":
                False,
        },
    ]

    dataframe = pd.DataFrame(
        rows
    )

    dataframe[
        "Model_Version"
    ] = MODEL_VERSION

    return dataframe


# ============================================================
# DATABASE
# ============================================================

def write_database(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    occurrence_metrics: pd.DataFrame,
    occurrence_behavior: pd.DataFrame,
    comparison: pd.DataFrame,
    config: pd.DataFrame,
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
            "metrics_df",
            metrics,
        )

        connection.register(
            "occurrence_df",
            occurrence_metrics,
        )

        connection.register(
            "occurrence_behavior_df",
            occurrence_behavior,
        )

        connection.register(
            "comparison_df",
            comparison,
        )

        connection.register(
            "config_df",
            config,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_sparse_model_predictions AS
            SELECT *
            FROM predictions_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_sparse_model_metrics AS
            SELECT *
            FROM metrics_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_occurrence_metrics AS
            SELECT *
            FROM occurrence_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_occurrence_by_behavior AS
            SELECT *
            FROM occurrence_behavior_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_all_model_comparison AS
            SELECT *
            FROM comparison_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_sparse_model_config AS
            SELECT *
            FROM config_df
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
    print("POISSON AND HURDLE ANNUAL FORECAST MODELS")
    print("=" * 104)
    print()

    supervised = (
        load_supervised()
    )

    benchmark_predictions = (
        load_benchmark_predictions()
    )

    prediction_frames = []
    occurrence_frames = []

    for fold in FOLDS:

        predictions, occurrence = (
            run_fold(
                supervised,
                fold,
            )
        )

        prediction_frames.append(
            predictions
        )

        occurrence_frames.append(
            occurrence
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    occurrence_metrics = pd.concat(
        occurrence_frames,
        ignore_index=True,
    )

    validate_predictions(
        predictions
    )

    print()
    print(
        "Prediction structural validation: PASS"
    )

    metrics = build_metrics(
        predictions
    )

    occurrence_behavior = (
        build_occurrence_by_behavior(
            predictions
        )
    )

    comparison = (
        build_all_model_comparison(
            benchmark_predictions,
            predictions,
        )
    )

    config = build_model_config()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    predictions.to_parquet(
        OUTPUT_DIR
        / "forecast_sparse_model_predictions.parquet",
        index=False,
        engine="pyarrow",
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "forecast_sparse_model_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics.to_csv(
        OUTPUT_DIR
        / "forecast_sparse_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_metrics.to_csv(
        OUTPUT_DIR
        / "forecast_occurrence_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    occurrence_behavior.to_csv(
        OUTPUT_DIR
        / "forecast_occurrence_by_behavior.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "forecast_all_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config.to_csv(
        OUTPUT_DIR
        / "forecast_sparse_model_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        predictions=predictions,
        metrics=metrics,
        occurrence_metrics=occurrence_metrics,
        occurrence_behavior=occurrence_behavior,
        comparison=comparison,
        config=config,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 104)
    print("NEW MODEL POINT-FORECAST PERFORMANCE")
    print("-" * 104)

    print(
        metrics.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 104)
    print("ALL SIX MODELS — OOS COMPARISON")
    print("-" * 104)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 104)
    print("HURDLE OCCURRENCE CLASSIFIER")
    print("-" * 104)

    print(
        occurrence_metrics.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 104)
    print("COMBINED OCCURRENCE PERFORMANCE BY AS-OF BEHAVIOR")
    print("-" * 104)

    combined_occurrence = (
        occurrence_behavior.loc[
            occurrence_behavior[
                "Fold_Name"
            ]
            .eq(
                COMBINED_OOS_LABEL
            )
        ]
    )

    print(
        combined_occurrence.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 104)
    print("GOVERNANCE")
    print("-" * 104)

    print(
        "No random train/test split used."
    )

    holdout_years = "/".join(
        str(
            fold["Test_Year"]
        )
        for fold in FOLDS
    )

    print(
        f"No {holdout_years} "
        "holdout-driven hyperparameter "
        "tuning performed."
    )

    print(
        "Occurrence threshold of 0.50 "
        "was predefined for diagnostic "
        "precision/recall/F1 only."
    )

    print(
        "Average Precision evaluates "
        "occurrence probabilities without "
        "selecting a classification threshold."
    )

    print(
        "Hurdle point forecast is "
        "probability × conditional positive "
        "quantity."
    )

    print(
        "No model is promoted to production "
        "by this script."
    )

    print()
    print("=" * 104)
    print(
        "SPARSE-DEMAND MODELING COMPLETE"
    )
    print("=" * 104)


if __name__ == "__main__":
    main()