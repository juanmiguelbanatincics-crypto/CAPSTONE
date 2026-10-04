from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn.metrics import mean_pinball_loss

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
    / "quantile_calibration_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

CALIBRATION_VERSION = (
    "TEMPORAL_QUANTILE_CALIBRATION_V1"
)

EVALUATION_FOLDS = (
    EVALUATION_CONTEXT.folds
)

if len(EVALUATION_FOLDS) < 2:
    raise ValueError(
        "Temporal quantile calibration "
        "requires at least two "
        "chronological backtest folds."
    )

CALIBRATION_FOLD_SPEC = (
    EVALUATION_FOLDS[-2]
)

VALIDATION_FOLD_SPEC = (
    EVALUATION_FOLDS[-1]
)

CALIBRATION_FOLD = (
    CALIBRATION_FOLD_SPEC["Fold_Name"]
)

VALIDATION_FOLD = (
    VALIDATION_FOLD_SPEC["Fold_Name"]
)

CALIBRATION_YEAR = int(
    CALIBRATION_FOLD_SPEC["Test_Year"]
)

VALIDATION_YEAR = int(
    VALIDATION_FOLD_SPEC["Test_Year"]
)

PLANNING_YEAR = int(
    RUNTIME_CONTEXT.planning_year
)

MIN_SEGMENT_CALIBRATION_ROWS = 100

QUANTILES = {
    "P50": 0.50,
    "P80": 0.80,
    "P90": 0.90,
}


# ============================================================
# LOAD OOS QUANTILE PREDICTIONS
# ============================================================

def load_predictions() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            """
            SELECT *
            FROM forecast_quantile_predictions
            ORDER BY Fold_Name, Medicine_Key
            """
        ).df()

    finally:
        connection.close()

    return dataframe


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(
    dataframe: pd.DataFrame,
) -> None:

    expected_folds = {
        fold["Fold_Name"]
        for fold in EVALUATION_FOLDS
    }

    actual_folds = set(
        dataframe[
            "Fold_Name"
        ]
        .unique()
        .tolist()
    )

    if actual_folds != expected_folds:

        raise ValueError(
            "Unexpected quantile folds. "
            f"Expected {sorted(expected_folds)}; "
            f"found {sorted(actual_folds)}."
        )

    for fold in EVALUATION_FOLDS:

        fold_name = fold[
            "Fold_Name"
        ]

        test_year = int(
            fold["Test_Year"]
        )

        fold_df = dataframe.loc[
            dataframe[
                "Fold_Name"
            ].eq(
                fold_name
            )
        ]

        if fold_df.empty:

            raise ValueError(
                f"{fold_name}: no prediction rows."
            )

        if (
            fold_df[
                "Medicine_Key"
            ]
            .duplicated()
            .any()
        ):

            raise ValueError(
                f"{fold_name}: duplicate "
                "Medicine_Key."
            )

        actual_target_years = set(
            fold_df[
                "Target_Year"
            ]
            .unique()
            .tolist()
        )

        if actual_target_years != {
            test_year
        }:

            raise ValueError(
                f"{fold_name}: expected target "
                f"year {test_year}; found "
                f"{actual_target_years}."
            )

    required_columns = [
        "Medicine_Key",
        "Fold_Name",
        "Target_Year",
        "Behavior_Segment_AsOf",
        "Target_Annual_Issuance",
        "P50",
        "P80",
        "P90",
    ]

    missing = [
        column
        for column
        in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )


# ============================================================
# FINITE-SAMPLE EMPIRICAL QUANTILE
# ============================================================

def calibration_quantile_level(
    n: int,
    quantile: float,
) -> float:

    if n <= 0:
        raise ValueError(
            "Calibration sample is empty."
        )

    # Slightly conservative finite-sample
    # quantile index.
    level = (
        np.ceil(
            (n + 1)
            * quantile
        )
        /
        n
    )

    return float(
        min(
            1.0,
            level,
        )
    )


def empirical_correction(
    residuals: pd.Series,
    quantile: float,
) -> tuple[
    float,
    float,
]:

    values = (
        pd.to_numeric(
            residuals,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    n = len(values)

    if n == 0:
        raise ValueError(
            "No residuals available "
            "for calibration."
        )

    level = calibration_quantile_level(
        n,
        quantile,
    )

    correction = float(
        np.quantile(
            values,
            level,
            method="higher",
        )
    )

    return (
        correction,
        level,
    )


# ============================================================
# BUILD TEMPORAL CALIBRATION CORRECTIONS
# ============================================================

def build_corrections(
    calibration: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # --------------------------------------------------------
    # GLOBAL FALLBACK
    # --------------------------------------------------------

    for label, quantile in (
        QUANTILES.items()
    ):

        residual = (
            calibration[
                "Target_Annual_Issuance"
            ]
            -
            calibration[
                label
            ]
        )

        correction, level = (
            empirical_correction(
                residual,
                quantile,
            )
        )

        rows.append(
            {
                "Scope":
                    "GLOBAL",

                "Behavior_Segment_AsOf":
                    "ALL",

                "Quantile_Label":
                    label,

                "Nominal_Quantile":
                    quantile,

                "Calibration_Rows":
                    len(calibration),

                "Empirical_Quantile_Level":
                    level,

                "Residual_Correction":
                    correction,

                "Uses_Global_Fallback":
                    False,

                "Calibration_Fold":
                    CALIBRATION_FOLD,

                "Calibration_Version":
                    CALIBRATION_VERSION,
            }
        )

    # --------------------------------------------------------
    # BEHAVIOR-SPECIFIC
    # --------------------------------------------------------

    for segment, segment_df in (
        calibration.groupby(
            "Behavior_Segment_AsOf",
            dropna=False,
            sort=True,
        )
    ):

        segment_name = (
            str(segment)
            if pd.notna(segment)
            else "MISSING"
        )

        for label, quantile in (
            QUANTILES.items()
        ):

            if (
                len(segment_df)
                >=
                MIN_SEGMENT_CALIBRATION_ROWS
            ):

                residual = (
                    segment_df[
                        "Target_Annual_Issuance"
                    ]
                    -
                    segment_df[
                        label
                    ]
                )

                correction, level = (
                    empirical_correction(
                        residual,
                        quantile,
                    )
                )

                fallback = False

            else:

                global_row = next(
                    row
                    for row in rows
                    if (
                        row["Scope"]
                        == "GLOBAL"
                        and
                        row[
                            "Quantile_Label"
                        ]
                        == label
                    )
                )

                correction = (
                    global_row[
                        "Residual_Correction"
                    ]
                )

                level = (
                    global_row[
                        "Empirical_Quantile_Level"
                    ]
                )

                fallback = True

            rows.append(
                {
                    "Scope":
                        "BEHAVIOR",

                    "Behavior_Segment_AsOf":
                        segment_name,

                    "Quantile_Label":
                        label,

                    "Nominal_Quantile":
                        quantile,

                    "Calibration_Rows":
                        len(segment_df),

                    "Empirical_Quantile_Level":
                        level,

                    "Residual_Correction":
                        correction,

                    "Uses_Global_Fallback":
                        fallback,

                    "Calibration_Fold":
                        CALIBRATION_FOLD,

                    "Calibration_Version":
                        CALIBRATION_VERSION,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# APPLY CORRECTIONS TO VALIDATION FOLD
# ============================================================

def apply_calibration(
    validation: pd.DataFrame,
    corrections: pd.DataFrame,
) -> pd.DataFrame:

    result = validation.copy()

    global_corrections = {}

    for label in QUANTILES:

        row = (
            corrections.loc[
                corrections[
                    "Scope"
                ].eq("GLOBAL")
                &
                corrections[
                    "Quantile_Label"
                ].eq(label)
            ]
            .iloc[0]
        )

        global_corrections[
            label
        ] = float(
            row[
                "Residual_Correction"
            ]
        )

    for label in QUANTILES:

        behavior_rows = (
            corrections.loc[
                corrections[
                    "Scope"
                ].eq("BEHAVIOR")
                &
                corrections[
                    "Quantile_Label"
                ].eq(label)
            ]
            .copy()
        )

        correction_map = dict(
            zip(
                behavior_rows[
                    "Behavior_Segment_AsOf"
                ],
                behavior_rows[
                    "Residual_Correction"
                ],
            )
        )

        segment_text = (
            result[
                "Behavior_Segment_AsOf"
            ]
            .astype("string")
            .fillna("MISSING")
        )

        result[
            f"{label}_Calibration_Correction"
        ] = (
            segment_text
            .map(
                correction_map
            )
            .fillna(
                global_corrections[
                    label
                ]
            )
            .astype(float)
        )

        result[
            f"Calibrated_{label}"
        ] = np.clip(
            result[
                label
            ]
            +
            result[
                f"{label}_Calibration_Correction"
            ],
            0,
            None,
        )

    # --------------------------------------------------------
    # ENFORCE MONOTONICITY
    # --------------------------------------------------------

    calibrated_matrix = (
        result[
            [
                "Calibrated_P50",
                "Calibrated_P80",
                "Calibrated_P90",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    monotonic = np.maximum.accumulate(
        calibrated_matrix,
        axis=1,
    )

    result[
        "Calibrated_P50"
    ] = monotonic[:, 0]

    result[
        "Calibrated_P80"
    ] = monotonic[:, 1]

    result[
        "Calibrated_P90"
    ] = monotonic[:, 2]

    result[
        "Calibration_Source"
    ] = (
        f"{CALIBRATION_YEAR}_"
        "OOS_BEHAVIOR_RESIDUALS"
    )

    result[
        "Calibration_Version"
    ] = CALIBRATION_VERSION

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
    forecast_column: str,
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
            forecast_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    coverage = float(
        np.mean(
            actual
            <=
            forecast
        )
    )

    pinball = float(
        mean_pinball_loss(
            actual,
            forecast,
            alpha=quantile,
        )
    )

    positive = (
        actual > 0
    )

    if positive.any():

        positive_coverage = float(
            np.mean(
                actual[
                    positive
                ]
                <=
                forecast[
                    positive
                ]
            )
        )

        positive_pinball = float(
            mean_pinball_loss(
                actual[
                    positive
                ],
                forecast[
                    positive
                ],
                alpha=quantile,
            )
        )

    else:

        positive_coverage = np.nan
        positive_pinball = np.nan

    return {
        "Rows":
            len(dataframe),

        "Positive_Targets":
            int(
                positive.sum()
            ),

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

        "Absolute_Coverage_Error_Pct_Points":
            abs(
                (
                    coverage
                    -
                    quantile
                )
                * 100
            ),

        "Mean_Pinball_Loss":
            pinball,

        "Positive_Target_Coverage_Pct":
            (
                positive_coverage
                * 100
            ),

        "Positive_Target_Pinball_Loss":
            positive_pinball,
    }


# ============================================================
# RAW VS TEMPORALLY CALIBRATED
# ============================================================

def build_validation_summary(
    calibrated: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for label, quantile in (
        QUANTILES.items()
    ):

        raw_metrics = (
            calculate_metrics(
                calibrated,
                label,
                quantile,
            )
        )

        calibrated_metrics = (
            calculate_metrics(
                calibrated,
                f"Calibrated_{label}",
                quantile,
            )
        )

        rows.append(
            {
                "Validation_Fold":
                    VALIDATION_FOLD,

                "Quantile_Label":
                    label,

                "Nominal_Quantile":
                    quantile,

                "Forecast_Type":
                    "RAW",

                **raw_metrics,
            }
        )

        rows.append(
            {
                "Validation_Fold":
                    VALIDATION_FOLD,

                "Quantile_Label":
                    label,

                "Nominal_Quantile":
                    quantile,

                "Forecast_Type":
                    "TEMPORALLY_CALIBRATED",

                **calibrated_metrics,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return summary


# ============================================================
# BY-BEHAVIOR VALIDATION
# ============================================================

def build_behavior_validation(
    calibrated: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for segment, segment_df in (
        calibrated.groupby(
            "Behavior_Segment_AsOf",
            dropna=False,
            sort=True,
        )
    ):

        segment_name = (
            str(segment)
            if pd.notna(segment)
            else "MISSING"
        )

        for label, quantile in (
            QUANTILES.items()
        ):

            for forecast_type, column in [
                (
                    "RAW",
                    label,
                ),
                (
                    "TEMPORALLY_CALIBRATED",
                    f"Calibrated_{label}",
                ),
            ]:

                metrics = (
                    calculate_metrics(
                        segment_df,
                        column,
                        quantile,
                    )
                )

                rows.append(
                    {
                        "Validation_Fold":
                            VALIDATION_FOLD,

                        "Behavior_Segment_AsOf":
                            segment_name,

                        "Quantile_Label":
                            label,

                        "Nominal_Quantile":
                            quantile,

                        "Forecast_Type":
                            forecast_type,

                        **metrics,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

def build_comparison(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    raw = (
        summary.loc[
            summary[
                "Forecast_Type"
            ].eq("RAW")
        ]
        .set_index(
            "Quantile_Label"
        )
    )

    calibrated = (
        summary.loc[
            summary[
                "Forecast_Type"
            ]
            .eq(
                "TEMPORALLY_CALIBRATED"
            )
        ]
        .set_index(
            "Quantile_Label"
        )
    )

    rows = []

    for label in QUANTILES:

        raw_row = raw.loc[
            label
        ]

        cal_row = calibrated.loc[
            label
        ]

        rows.append(
            {
                "Quantile_Label":
                    label,

                "Nominal_Quantile":
                    QUANTILES[
                        label
                    ],

                "Raw_Coverage_Pct":
                    raw_row[
                        "Empirical_Coverage_Pct"
                    ],

                "Calibrated_Coverage_Pct":
                    cal_row[
                        "Empirical_Coverage_Pct"
                    ],

                "Raw_Absolute_Coverage_Error":
                    raw_row[
                        "Absolute_Coverage_Error_Pct_Points"
                    ],

                "Calibrated_Absolute_Coverage_Error":
                    cal_row[
                        "Absolute_Coverage_Error_Pct_Points"
                    ],

                "Coverage_Error_Improvement":
                    (
                        raw_row[
                            "Absolute_Coverage_Error_Pct_Points"
                        ]
                        -
                        cal_row[
                            "Absolute_Coverage_Error_Pct_Points"
                        ]
                    ),

                "Raw_Pinball":
                    raw_row[
                        "Mean_Pinball_Loss"
                    ],

                "Calibrated_Pinball":
                    cal_row[
                        "Mean_Pinball_Loss"
                    ],

                "Pinball_Improvement":
                    (
                        raw_row[
                            "Mean_Pinball_Loss"
                        ]
                        -
                        cal_row[
                            "Mean_Pinball_Loss"
                        ]
                    ),

                "Calibration_Improved_Coverage_Error":
                    bool(
                        cal_row[
                            "Absolute_Coverage_Error_Pct_Points"
                        ]
                        <
                        raw_row[
                            "Absolute_Coverage_Error_Pct_Points"
                        ]
                    ),

                "Calibration_Improved_Pinball":
                    bool(
                        cal_row[
                            "Mean_Pinball_Loss"
                        ]
                        <
                        raw_row[
                            "Mean_Pinball_Loss"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_calibrated(
    dataframe: pd.DataFrame,
) -> None:

    if dataframe.empty:

        raise ValueError(
            "Temporal calibration validation "
            "population is empty."
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate Medicine_Key in "
            "calibration validation population."
        )

    if not (
        dataframe[
            "Fold_Name"
        ]
        .eq(
            VALIDATION_FOLD
        )
        .all()
    ):

        raise ValueError(
            "Unexpected validation fold in "
            "calibrated predictions."
        )

    if not (
        dataframe[
            "Target_Year"
        ]
        .eq(
            VALIDATION_YEAR
        )
        .all()
    ):

        raise ValueError(
            "Unexpected target year in "
            "calibrated predictions."
        )

    if (
        dataframe[
            "Calibrated_P50"
        ]
        .gt(
            dataframe[
                "Calibrated_P80"
            ]
        )
        .any()
    ):

        raise ValueError(
            "Calibrated P50 > P80."
        )

    if (
        dataframe[
            "Calibrated_P80"
        ]
        .gt(
            dataframe[
                "Calibrated_P90"
            ]
        )
        .any()
    ):

        raise ValueError(
            "Calibrated P80 > P90."
        )

    if (
        dataframe[
            [
                "Calibrated_P50",
                "Calibrated_P80",
                "Calibrated_P90",
            ]
        ]
        .lt(0)
        .any()
        .any()
    ):

        raise ValueError(
            "Negative calibrated "
            "quantile detected."
        )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    corrections: pd.DataFrame,
    calibrated: pd.DataFrame,
    summary: pd.DataFrame,
    behavior: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "corrections_df",
            corrections,
        )

        connection.register(
            "calibrated_df",
            calibrated,
        )

        connection.register(
            "summary_df",
            summary,
        )

        connection.register(
            "behavior_df",
            behavior,
        )

        connection.register(
            "comparison_df",
            comparison,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_calibration_corrections AS
            SELECT *
            FROM corrections_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            forecast_quantile_calibrated_{VALIDATION_YEAR} AS
            SELECT *
            FROM calibrated_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_calibration_validation AS
            SELECT *
            FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_calibration_by_behavior AS
            SELECT *
            FROM behavior_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_quantile_calibration_comparison AS
            SELECT *
            FROM comparison_df
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
    print("TEMPORAL QUANTILE CALIBRATION VALIDATION")
    print("=" * 108)
    print()

    predictions = (
        load_predictions()
    )

    validate_input(
        predictions
    )

    print(
        "Quantile prediction input: PASS"
    )

    calibration = (
        predictions.loc[
            predictions[
                "Fold_Name"
            ].eq(
                CALIBRATION_FOLD
            )
        ]
        .copy()
    )

    validation = (
        predictions.loc[
            predictions[
                "Fold_Name"
            ].eq(
                VALIDATION_FOLD
            )
        ]
        .copy()
    )

    print(
        "Calibration period:",
        f"{CALIBRATION_YEAR} OOS",
    )

    print(
        "Validation period:",
        f"{VALIDATION_YEAR} OOS",
    )

    corrections = (
        build_corrections(
            calibration
        )
    )

    calibrated = (
        apply_calibration(
            validation,
            corrections,
        )
    )

    validate_calibrated(
        calibrated
    )

    print(
        "Calibrated forecast "
        "structural validation: PASS"
    )

    summary = (
        build_validation_summary(
            calibrated
        )
    )

    behavior = (
        build_behavior_validation(
            calibrated
        )
    )

    comparison = (
        build_comparison(
            summary
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    corrections.to_csv(
        OUTPUT_DIR
        / f"quantile_calibration_corrections_{CALIBRATION_YEAR}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibrated.to_parquet(
        OUTPUT_DIR
        / f"quantile_calibrated_predictions_{VALIDATION_YEAR}.parquet",
        index=False,
        engine="pyarrow",
    )

    calibrated.to_csv(
        OUTPUT_DIR
        / f"quantile_calibrated_predictions_{VALIDATION_YEAR}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / f"quantile_calibration_validation_{VALIDATION_YEAR}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    behavior.to_csv(
        OUTPUT_DIR
        / f"quantile_calibration_by_behavior_{VALIDATION_YEAR}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        OUTPUT_DIR
        / f"quantile_calibration_comparison_{VALIDATION_YEAR}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        corrections=corrections,
        calibrated=calibrated,
        summary=summary,
        behavior=behavior,
        comparison=comparison,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 108)
    print(f"{CALIBRATION_YEAR} CALIBRATION CORRECTIONS")
    print("-" * 108)

    print(
        corrections.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print(f"{VALIDATION_YEAR} RAW VS TEMPORALLY CALIBRATED")
    print("-" * 108)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print(f"{VALIDATION_YEAR} CALIBRATION BY AS-OF BEHAVIOR")
    print("-" * 108)

    behavior_display = (
        behavior.loc[
            behavior[
                "Quantile_Label"
            ]
            .isin(
                [
                    "P80",
                    "P90",
                ]
            )
        ]
        .sort_values(
            [
                "Behavior_Segment_AsOf",
                "Quantile_Label",
                "Forecast_Type",
            ]
        )
    )

    print(
        behavior_display[
            [
                "Behavior_Segment_AsOf",
                "Quantile_Label",
                "Forecast_Type",
                "Rows",
                "Positive_Targets",
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
    print("INTERPRETATION")
    print("-" * 108)

    print(
        f"{CALIBRATION_YEAR} is used only "
        "to estimate residual calibration "
        "corrections."
    )

    print(
        f"{VALIDATION_YEAR} is used as the "
        "later temporal validation period "
        "for those corrections."
    )

    print(
        f"No {VALIDATION_YEAR} outcome was "
        "used to calculate the corrections "
        f"applied to {VALIDATION_YEAR}."
    )

    print(
        "Exact nominal coverage is not "
        "guaranteed for discrete, zero-heavy "
        "annual issuance."
    )

    print(
        "Calibration should be judged using "
        "coverage behavior together with "
        "pinball loss, especially for P80/P90."
    )

    print(
        f"No {PLANNING_YEAR} uncertainty "
        "policy is promoted automatically "
        "by this script."
    )

    print()
    print("=" * 108)
    print(
        "TEMPORAL QUANTILE CALIBRATION "
        "VALIDATION COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()