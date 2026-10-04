from __future__ import annotations

from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

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
    / "bootstrap_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BOOTSTRAP GOVERNANCE
# ============================================================

N_BOOTSTRAP = 1000
RANDOM_STATE = 42

BOOTSTRAP_VERSION = (
    "FORECAST_PAIRED_BOOTSTRAP_V1"
)

REFERENCE_MODEL = "NAIVE_LAG1"

EXPECTED_MODELS = [
    "NAIVE_LAG1",
    "MOVING_AVERAGE_2",
    "ELASTIC_NET_GLOBAL",
    "HIST_GRADIENT_BOOSTING_GLOBAL",
    "HIST_GRADIENT_BOOSTING_POISSON",
    "HURDLE_HGB_EXPECTED",
]

FOLDS = EVALUATION_CONTEXT.folds

COMBINED_OOS_LABEL = (
    EVALUATION_CONTEXT.combined_oos_label
)

EVALUATION_POPULATIONS = (
    EVALUATION_CONTEXT.evaluation_populations
)



# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        benchmark = connection.execute(
            """
            SELECT
                Fold_Name,
                Model,
                Medicine_Key,
                Target_Year,
                Target_Annual_Issuance,
                Prediction
            FROM forecast_benchmark_predictions
            """
        ).df()

        sparse = connection.execute(
            """
            SELECT
                Fold_Name,
                Model,
                Medicine_Key,
                Target_Year,
                Target_Annual_Issuance,
                Prediction
            FROM forecast_sparse_model_predictions
            """
        ).df()

    finally:

        connection.close()

    predictions = pd.concat(
        [
            benchmark,
            sparse,
        ],
        ignore_index=True,
    )

    predictions[
        "Absolute_Error"
    ] = (
        predictions[
            "Prediction"
        ]
        -
        predictions[
            "Target_Annual_Issuance"
        ]
    ).abs()

    predictions[
        "Squared_Error"
    ] = (
        (
            predictions[
                "Prediction"
            ]
            -
            predictions[
                "Target_Annual_Issuance"
            ]
        )
        ** 2
    )

    return predictions


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_predictions(
    predictions: pd.DataFrame,
) -> None:

    actual_models = set(
        predictions[
            "Model"
        ]
        .unique()
        .tolist()
    )

    if actual_models != set(
        EXPECTED_MODELS
    ):

        raise ValueError(
            "Unexpected model set.\n"
            f"Expected: {EXPECTED_MODELS}\n"
            f"Actual: {sorted(actual_models)}"
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
            "Duplicate OOS prediction "
            f"records detected: "
            f"{duplicate_count}"
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
            "Unexpected bootstrap fold set. "
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
                f"target year evidence "
                f"{observed_years}."
            )

        reference_keys = None
        reference_actual = None

        for model in EXPECTED_MODELS:

            model_df = (
                fold_df.loc[
                    fold_df[
                        "Model"
                    ].eq(
                        model
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
                    f"{fold_name} / {model}: "
                    "no prediction rows."
                )

            if model_df[
                "Medicine_Key"
            ].duplicated().any():

                raise ValueError(
                    f"{fold_name} / {model}: "
                    "duplicate medicine evidence."
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
                        f"{fold_name}: actual "
                        "target evidence differs "
                        "between models."
                    )


# ============================================================
# EVALUATION POPULATION
# ============================================================

def get_population(
    predictions: pd.DataFrame,
    population: str,
) -> pd.DataFrame:

    fold_names = {
        fold["Fold_Name"]
        for fold in FOLDS
    }

    if population in fold_names:

        result = predictions.loc[
            predictions[
                "Fold_Name"
            ].eq(
                population
            )
        ].copy()

    elif (
        population
        == COMBINED_OOS_LABEL
    ):

        result = predictions.copy()

    else:

        raise ValueError(
            f"Unknown population: "
            f"{population}"
        )

    return result


# ============================================================
# AGGREGATE TO MEDICINE LEVEL
# ============================================================

def aggregate_by_medicine(
    population_df: pd.DataFrame,
) -> pd.DataFrame:

    aggregated = (
        population_df
        .groupby(
            [
                "Model",
                "Medicine_Key",
            ],
            as_index=False,
        )
        .agg(
            Row_Count=(
                "Target_Year",
                "size",
            ),
            Actual_Total=(
                "Target_Annual_Issuance",
                "sum",
            ),
            Forecast_Total=(
                "Prediction",
                "sum",
            ),
            Absolute_Error_Total=(
                "Absolute_Error",
                "sum",
            ),
            Squared_Error_Total=(
                "Squared_Error",
                "sum",
            ),
        )
    )

    return aggregated


# ============================================================
# ALIGN MODEL ARRAYS
# ============================================================

def build_aligned_arrays(
    aggregated: pd.DataFrame,
) -> tuple[
    list[str],
    dict,
]:

    medicine_keys = sorted(
        aggregated[
            "Medicine_Key"
        ]
        .unique()
        .tolist()
    )

    if not medicine_keys:

        raise ValueError(
            "Bootstrap population "
            "contains no medicines."
        )

    arrays = {}

    reference_actual = None
    reference_rows = None

    for model in EXPECTED_MODELS:

        model_df = (
            aggregated.loc[
                aggregated[
                    "Model"
                ].eq(
                    model
                )
            ]
            .set_index(
                "Medicine_Key"
            )
            .reindex(
                medicine_keys
            )
        )

        if model_df.isna().all(
            axis=1
        ).any():

            raise ValueError(
                f"Alignment failure for "
                f"{model}."
            )

        arrays[model] = {
            "rows":
                model_df[
                    "Row_Count"
                ]
                .to_numpy(
                    dtype=float
                ),

            "actual":
                model_df[
                    "Actual_Total"
                ]
                .to_numpy(
                    dtype=float
                ),

            "forecast":
                model_df[
                    "Forecast_Total"
                ]
                .to_numpy(
                    dtype=float
                ),

            "absolute_error":
                model_df[
                    "Absolute_Error_Total"
                ]
                .to_numpy(
                    dtype=float
                ),

            "squared_error":
                model_df[
                    "Squared_Error_Total"
                ]
                .to_numpy(
                    dtype=float
                ),
        }

        if reference_actual is None:

            reference_actual = (
                arrays[model][
                    "actual"
                ].copy()
            )

            reference_rows = (
                arrays[model][
                    "rows"
                ].copy()
            )

        else:

            if not np.allclose(
                arrays[model][
                    "actual"
                ],
                reference_actual,
                atol=0,
                rtol=0,
            ):

                raise ValueError(
                    "Actual values differ "
                    "between models."
                )

            if not np.allclose(
                arrays[model][
                    "rows"
                ],
                reference_rows,
                atol=0,
                rtol=0,
            ):

                raise ValueError(
                    "Observation counts differ "
                    "between models."
                )

    return (
        medicine_keys,
        arrays,
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def metrics_from_arrays(
    arrays: dict,
    indices: np.ndarray,
) -> dict:

    rows = float(
        arrays[
            "rows"
        ][indices]
        .sum()
    )

    actual_total = float(
        arrays[
            "actual"
        ][indices]
        .sum()
    )

    forecast_total = float(
        arrays[
            "forecast"
        ][indices]
        .sum()
    )

    absolute_error_total = float(
        arrays[
            "absolute_error"
        ][indices]
        .sum()
    )

    squared_error_total = float(
        arrays[
            "squared_error"
        ][indices]
        .sum()
    )

    mae = (
        absolute_error_total
        /
        rows
    )

    rmse = float(
        np.sqrt(
            squared_error_total
            /
            rows
        )
    )

    if actual_total > 0:

        wape = (
            absolute_error_total
            /
            actual_total
            * 100
        )

        bias = (
            (
                forecast_total
                -
                actual_total
            )
            /
            actual_total
            * 100
        )

    else:

        wape = np.nan
        bias = np.nan

    return {
        "MAE":
            mae,

        "RMSE":
            rmse,

        "WAPE_Pct":
            wape,

        "Forecast_Bias_Pct":
            bias,

        "Absolute_Bias_Pct":
            abs(bias)
            if pd.notna(bias)
            else np.nan,

        "Actual_Total":
            actual_total,

        "Forecast_Total":
            forecast_total,
    }


# ============================================================
# OBSERVED METRICS
# ============================================================

def calculate_observed_metrics(
    arrays: dict,
) -> pd.DataFrame:

    rows = []

    n_medicines = len(
        arrays[
            EXPECTED_MODELS[0]
        ][
            "rows"
        ]
    )

    full_indices = np.arange(
        n_medicines
    )

    for model in EXPECTED_MODELS:

        metrics = metrics_from_arrays(
            arrays[model],
            full_indices,
        )

        rows.append(
            {
                "Model":
                    model,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def run_bootstrap_population(
    population: str,
    arrays: dict,
    seed_offset: int,
) -> pd.DataFrame:

    rng = np.random.default_rng(
        RANDOM_STATE
        + seed_offset
    )

    records = []

    n_medicines = len(
        arrays[
            EXPECTED_MODELS[0]
        ][
            "rows"
        ]
    )

    if n_medicines < 1:
        raise ValueError(
            "Bootstrap population "
            "contains no medicines."
        )

    for bootstrap_id in range(
        1,
        N_BOOTSTRAP + 1,
    ):

        # Same sampled Medicine_Key positions
        # are used for every model.
        #
        # For the combined population each
        # sampled medicine contributes both
        # 2024 and 2025 holdout observations.

        sampled_indices = (
            rng.integers(
                low=0,
                high=n_medicines,
                size=n_medicines,
            )
        )

        replicate_metrics = {}

        for model in EXPECTED_MODELS:

            metrics = metrics_from_arrays(
                arrays[model],
                sampled_indices,
            )

            replicate_metrics[
                model
            ] = metrics

        # WAPE winner in this bootstrap draw
        wape_values = {
            model:
                replicate_metrics[
                    model
                ][
                    "WAPE_Pct"
                ]
            for model in EXPECTED_MODELS
        }

        minimum_wape = min(
            wape_values.values()
        )

        for model in EXPECTED_MODELS:

            metrics = replicate_metrics[
                model
            ]

            records.append(
                {
                    "Evaluation_Population":
                        population,

                    "Bootstrap_ID":
                        bootstrap_id,

                    "Model":
                        model,

                    **metrics,

                    "WAPE_Winner":
                        bool(
                            np.isclose(
                                metrics[
                                    "WAPE_Pct"
                                ],
                                minimum_wape,
                                atol=1e-12,
                                rtol=0,
                            )
                        ),

                    "Bootstrap_Unit":
                        "Medicine_Key",

                    "Bootstrap_Version":
                        BOOTSTRAP_VERSION,
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# MODEL BOOTSTRAP INTERVALS
# ============================================================

def percentile_interval(
    series: pd.Series,
) -> tuple[
    float,
    float,
]:

    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    lower, upper = np.percentile(
        values,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(lower),
        float(upper),
    )


def build_model_summary(
    observed: pd.DataFrame,
    bootstrap: pd.DataFrame,
    population: str,
) -> pd.DataFrame:

    rows = []

    for model in EXPECTED_MODELS:

        observed_row = (
            observed.loc[
                observed[
                    "Model"
                ].eq(
                    model
                )
            ]
            .iloc[0]
        )

        boot = (
            bootstrap.loc[
                bootstrap[
                    "Model"
                ].eq(
                    model
                )
            ]
        )

        wape_low, wape_high = (
            percentile_interval(
                boot[
                    "WAPE_Pct"
                ]
            )
        )

        mae_low, mae_high = (
            percentile_interval(
                boot[
                    "MAE"
                ]
            )
        )

        rmse_low, rmse_high = (
            percentile_interval(
                boot[
                    "RMSE"
                ]
            )
        )

        bias_low, bias_high = (
            percentile_interval(
                boot[
                    "Forecast_Bias_Pct"
                ]
            )
        )

        rows.append(
            {
                "Evaluation_Population":
                    population,

                "Model":
                    model,

                "Observed_WAPE_Pct":
                    float(
                        observed_row[
                            "WAPE_Pct"
                        ]
                    ),

                "WAPE_Bootstrap_Lower95":
                    wape_low,

                "WAPE_Bootstrap_Upper95":
                    wape_high,

                "Observed_MAE":
                    float(
                        observed_row[
                            "MAE"
                        ]
                    ),

                "MAE_Bootstrap_Lower95":
                    mae_low,

                "MAE_Bootstrap_Upper95":
                    mae_high,

                "Observed_RMSE":
                    float(
                        observed_row[
                            "RMSE"
                        ]
                    ),

                "RMSE_Bootstrap_Lower95":
                    rmse_low,

                "RMSE_Bootstrap_Upper95":
                    rmse_high,

                "Observed_Bias_Pct":
                    float(
                        observed_row[
                            "Forecast_Bias_Pct"
                        ]
                    ),

                "Bias_Bootstrap_Lower95":
                    bias_low,

                "Bias_Bootstrap_Upper95":
                    bias_high,

                "WAPE_Win_Probability":
                    float(
                        boot[
                            "WAPE_Winner"
                        ]
                        .mean()
                    ),

                "Bootstrap_Replications":
                    N_BOOTSTRAP,

                "Bootstrap_Unit":
                    "Medicine_Key",
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PAIRWISE COMPARISON
# ============================================================

def build_pairwise_comparison(
    observed: pd.DataFrame,
    bootstrap: pd.DataFrame,
    population: str,
) -> pd.DataFrame:

    rows = []

    metrics = [
        "WAPE_Pct",
        "MAE",
        "RMSE",
        "Absolute_Bias_Pct",
    ]

    observed_index = (
        observed.set_index(
            "Model"
        )
    )

    for model_a, model_b in combinations(
        EXPECTED_MODELS,
        2,
    ):

        a_boot = (
            bootstrap.loc[
                bootstrap[
                    "Model"
                ].eq(
                    model_a
                )
            ]
            .sort_values(
                "Bootstrap_ID"
            )
            .reset_index(
                drop=True
            )
        )

        b_boot = (
            bootstrap.loc[
                bootstrap[
                    "Model"
                ].eq(
                    model_b
                )
            ]
            .sort_values(
                "Bootstrap_ID"
            )
            .reset_index(
                drop=True
            )
        )

        if not np.array_equal(
            a_boot[
                "Bootstrap_ID"
            ].to_numpy(),
            b_boot[
                "Bootstrap_ID"
            ].to_numpy(),
        ):

            raise ValueError(
                "Paired bootstrap alignment "
                "failure."
            )

        for metric in metrics:

            if (
                metric
                == "Absolute_Bias_Pct"
            ):

                observed_a = abs(
                    float(
                        observed_index.loc[
                            model_a,
                            "Forecast_Bias_Pct",
                        ]
                    )
                )

                observed_b = abs(
                    float(
                        observed_index.loc[
                            model_b,
                            "Forecast_Bias_Pct",
                        ]
                    )
                )

            else:

                observed_a = float(
                    observed_index.loc[
                        model_a,
                        metric,
                    ]
                )

                observed_b = float(
                    observed_index.loc[
                        model_b,
                        metric,
                    ]
                )

            difference = (
                a_boot[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
                -
                b_boot[
                    metric
                ]
                .to_numpy(
                    dtype=float
                )
            )

            lower, upper = np.percentile(
                difference,
                [
                    2.5,
                    97.5,
                ],
            )

            probability_a_better = float(
                np.mean(
                    difference < 0
                )
            )

            if upper < 0:

                interpretation = (
                    "MODEL_A_LOWER_METRIC"
                )

            elif lower > 0:

                interpretation = (
                    "MODEL_B_LOWER_METRIC"
                )

            else:

                interpretation = (
                    "INTERVAL_INCLUDES_ZERO"
                )

            rows.append(
                {
                    "Evaluation_Population":
                        population,

                    "Model_A":
                        model_a,

                    "Model_B":
                        model_b,

                    "Metric":
                        metric,

                    "Observed_Difference_A_minus_B":
                        (
                            observed_a
                            -
                            observed_b
                        ),

                    "Difference_Lower95":
                        float(lower),

                    "Difference_Upper95":
                        float(upper),

                    "Probability_Model_A_Better":
                        probability_a_better,

                    "Bootstrap_Interpretation":
                        interpretation,

                    "Bootstrap_Replications":
                        N_BOOTSTRAP,

                    "Bootstrap_Unit":
                        "Medicine_Key",
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CONFIG
# ============================================================

def build_config() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "Bootstrap_Version":
                    BOOTSTRAP_VERSION,

                "Replications":
                    N_BOOTSTRAP,

                "Random_State":
                    RANDOM_STATE,

                "Resampling_Unit":
                    "Medicine_Key",

                "Combined_Years_Preserved":
                    True,

                "Paired_Across_Models":
                    True,

                "Interval_Method":
                    (
                        "95% percentile "
                        "bootstrap interval"
                    ),

                "Reference_Model":
                    REFERENCE_MODEL,

                "Notes":
                    (
                        (
                            "For combined OOS, "
                            f"both {FOLDS[0]['Test_Year']} "
                            f"and {FOLDS[-1]['Test_Year']} "
                            "records for a sampled "
                            "medicine remain together."
                        )
                        if len(FOLDS) == 2
                        else
                        (
                            "For combined OOS, all "
                            f"available {FOLDS[0]['Test_Year']}-"
                            f"{FOLDS[-1]['Test_Year']} "
                            "OOS records for a sampled "
                            "medicine remain together."
                        )
                    ),
            }
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    replicate_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    config: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "replicate_df",
            replicate_df,
        )

        connection.register(
            "model_summary_df",
            model_summary,
        )

        connection.register(
            "pairwise_df",
            pairwise,
        )

        connection.register(
            "config_df",
            config,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_bootstrap_replicates AS
            SELECT *
            FROM replicate_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_bootstrap_model_summary AS
            SELECT *
            FROM model_summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_bootstrap_pairwise AS
            SELECT *
            FROM pairwise_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_bootstrap_config AS
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

    print("=" * 108)
    print("CDH INVENTORY DSS")
    print("PAIRED MEDICINE-LEVEL FORECAST BOOTSTRAP")
    print("=" * 108)
    print()

    predictions = (
        load_predictions()
    )

    combined_year_range = (
        f"{FOLDS[0]['Test_Year']}-"
        f"{FOLDS[-1]['Test_Year']}"
    )

    validate_predictions(
        predictions
    )

    print(
        "Prediction input validation: PASS"
    )

    print(
        "Models frozen for comparison:",
        len(EXPECTED_MODELS),
    )

    print(
        "Bootstrap replications:",
        f"{N_BOOTSTRAP:,}",
    )

    print(
        "Bootstrap unit: Medicine_Key"
    )

    all_replicates = []
    all_model_summaries = []
    all_pairwise = []

    for population_index, population in enumerate(
        EVALUATION_POPULATIONS
    ):

        print()
        print("-" * 108)

        print(
            "Evaluation population:",
            population,
        )

        population_df = get_population(
            predictions,
            population,
        )

        aggregated = aggregate_by_medicine(
            population_df
        )

        medicine_keys, arrays = (
            build_aligned_arrays(
                aggregated
            )
        )

        print(
            "Unique medicines:",
            f"{len(medicine_keys):,}",
        )

        observed_row_count = set(
            aggregated[
                "Row_Count"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        if (
            population
            != COMBINED_OOS_LABEL
        ):

            if observed_row_count != {1}:

                raise ValueError(
                    f"{population}: each medicine "
                    "must contribute exactly one "
                    "OOS observation."
                )

        else:

            if (
                min(observed_row_count) < 1
                or
                max(observed_row_count) > len(FOLDS)
            ):

                raise ValueError(
                    f"{population}: combined "
                    "medicine observation counts "
                    "are outside the valid range."
                )

        observed = (
            calculate_observed_metrics(
                arrays
            )
        )

        bootstrap = (
            run_bootstrap_population(
                population=population,
                arrays=arrays,
                seed_offset=(
                    population_index
                    * 10000
                ),
            )
        )

        summary = (
            build_model_summary(
                observed=observed,
                bootstrap=bootstrap,
                population=population,
            )
        )

        pairwise = (
            build_pairwise_comparison(
                observed=observed,
                bootstrap=bootstrap,
                population=population,
            )
        )

        all_replicates.append(
            bootstrap
        )

        all_model_summaries.append(
            summary
        )

        all_pairwise.append(
            pairwise
        )

        print(
            "Bootstrap complete:",
            f"{N_BOOTSTRAP:,}",
            "paired replications",
        )

    replicate_df = pd.concat(
        all_replicates,
        ignore_index=True,
    )

    model_summary = pd.concat(
        all_model_summaries,
        ignore_index=True,
    )

    pairwise = pd.concat(
        all_pairwise,
        ignore_index=True,
    )

    config = build_config()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    replicate_df.to_parquet(
        OUTPUT_DIR
        / "forecast_bootstrap_replicates.parquet",
        index=False,
        engine="pyarrow",
    )

    model_summary.to_csv(
        OUTPUT_DIR
        / "forecast_bootstrap_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pairwise.to_csv(
        OUTPUT_DIR
        / "forecast_bootstrap_pairwise.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config.to_csv(
        OUTPUT_DIR
        / "forecast_bootstrap_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        replicate_df=replicate_df,
        model_summary=model_summary,
        pairwise=pairwise,
        config=config,
    )

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    combined_summary = (
        model_summary.loc[
            model_summary[
                "Evaluation_Population"
            ]
            .eq(
                COMBINED_OOS_LABEL
            )
        ]
        .sort_values(
            "Observed_WAPE_Pct"
        )
    )

    print()
    print("=" * 108)
    print(
        f"COMBINED {combined_year_range} MODEL "
        "BOOTSTRAP SUMMARY"
    )
    print("=" * 108)

    print(
        combined_summary[
            [
                "Model",
                "Observed_WAPE_Pct",
                "WAPE_Bootstrap_Lower95",
                "WAPE_Bootstrap_Upper95",
                "Observed_MAE",
                "Observed_RMSE",
                "Observed_Bias_Pct",
                "WAPE_Win_Probability",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # PAIRWISE AGAINST NAIVE
    # --------------------------------------------------------

    reference_comparison = (
        pairwise.loc[
            pairwise[
                "Evaluation_Population"
            ]
            .eq(
                COMBINED_OOS_LABEL
            )
            &
            pairwise[
                "Metric"
            ]
            .eq(
                "WAPE_Pct"
            )
            &
            (
                pairwise[
                    "Model_A"
                ]
                .eq(
                    REFERENCE_MODEL
                )
                |
                pairwise[
                    "Model_B"
                ]
                .eq(
                    REFERENCE_MODEL
                )
            )
        ]
        .copy()
    )

    print()
    print("=" * 108)
    print(
        "COMBINED WAPE — PAIRWISE "
        "COMPARISONS INVOLVING NAIVE"
    )
    print("=" * 108)

    print(
        reference_comparison[
            [
                "Model_A",
                "Model_B",
                "Observed_Difference_A_minus_B",
                "Difference_Lower95",
                "Difference_Upper95",
                "Probability_Model_A_Better",
                "Bootstrap_Interpretation",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # HURDLE VS NAIVE SPECIFIC
    # --------------------------------------------------------

    hurdle_naive = (
        pairwise.loc[
            pairwise[
                "Evaluation_Population"
            ]
            .eq(
                COMBINED_OOS_LABEL
            )
            &
            pairwise[
                "Metric"
            ]
            .isin(
                [
                    "WAPE_Pct",
                    "MAE",
                    "RMSE",
                    "Absolute_Bias_Pct",
                ]
            )
            &
            (
                (
                    pairwise[
                        "Model_A"
                    ]
                    .eq(
                        "NAIVE_LAG1"
                    )
                    &
                    pairwise[
                        "Model_B"
                    ]
                    .eq(
                        "HURDLE_HGB_EXPECTED"
                    )
                )
                |
                (
                    pairwise[
                        "Model_A"
                    ]
                    .eq(
                        "HURDLE_HGB_EXPECTED"
                    )
                    &
                    pairwise[
                        "Model_B"
                    ]
                    .eq(
                        "NAIVE_LAG1"
                    )
                )
            )
        ]
    )

    print()
    print("=" * 108)
    print(
        "HURDLE VS NAIVE — "
        "COMBINED OOS"
    )
    print("=" * 108)

    print(
        hurdle_naive[
            [
                "Model_A",
                "Model_B",
                "Metric",
                "Observed_Difference_A_minus_B",
                "Difference_Lower95",
                "Difference_Upper95",
                "Probability_Model_A_Better",
                "Bootstrap_Interpretation",
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
    print("INTERPRETATION RULE")
    print("-" * 108)

    print(
        "Pairwise difference is "
        "Model_A metric minus Model_B metric."
    )

    print(
        "For WAPE, MAE, RMSE, and "
        "absolute bias, LOWER is better."
    )

    print(
        "If the 95% paired-bootstrap "
        "difference interval includes zero, "
        "we will treat the performance "
        "difference as uncertain rather than "
        "claiming a stable winner."
    )

    print(
        "WAPE_Win_Probability is descriptive "
        "bootstrap stability, not a classical "
        "p-value."
    )

    print()
    print("=" * 108)
    print(
        "PAIRED BOOTSTRAP COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()