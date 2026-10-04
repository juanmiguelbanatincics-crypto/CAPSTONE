from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    from .runtime_context import load_runtime_context_from_env
except ImportError:
    from runtime_context import load_runtime_context_from_env


# ============================================================
# GOVERNED RUNTIME CONTEXT
# ============================================================

RUNTIME_CONTEXT = load_runtime_context_from_env()

PROJECT_ROOT = RUNTIME_CONTEXT.project_root
DATABASE_PATH = RUNTIME_CONTEXT.database_path

OUTPUT_DIR = (
    RUNTIME_CONTEXT.processed_dir
    / "forecasting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

HISTORICAL_YEARS = (
    RUNTIME_CONTEXT.historical_years
)

SUPERVISED_TARGET_YEARS = (
    RUNTIME_CONTEXT.supervised_target_years
)

SCORING_TARGET_YEAR = (
    RUNTIME_CONTEXT.planning_year
)

PERSISTENCE_THRESHOLD = 0.80

FEATURE_SET_VERSION = "FORECAST_FEATURES_V1"


# ============================================================
# LOAD DATA
# ============================================================

def load_fact() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        dataframe = connection.execute(
            """
            SELECT
                Medicine_Key,
                Year,
                Brand_Name,
                Generic_Name,
                Unit,
                Annual_Issuance,
                Issuance_Active,
                Beginning_Balance,
                Qty_in_Deliveries,
                Qty_in_Total_In,
                Qty_out_Total_Out,
                Ending_Balance,
                Average_cost,
                Cost_Valid,
                Data_Quality_Status
            FROM fact_inventory_annual
            ORDER BY Medicine_Key, Year
            """
        ).df()

    finally:
        connection.close()

    return dataframe


def load_abc() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        dataframe = connection.execute(
            """
            SELECT
                Medicine_Key,
                Year,
                ABC_Class
            FROM abc_classification_annual
            ORDER BY Medicine_Key, Year
            """
        ).df()

    finally:
        connection.close()

    return dataframe


# ============================================================
# STATISTICAL HELPERS
# ============================================================

def sample_sd(
    values: pd.Series,
) -> float:

    values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    if len(values) < 2:
        return np.nan

    return float(
        values.std(
            ddof=1
        )
    )


def trend_slope(
    years: np.ndarray,
    values: np.ndarray,
) -> float:

    if len(years) < 2:
        return np.nan

    return float(
        np.polyfit(
            years,
            values,
            1,
        )[0]
    )


# ============================================================
# CREATE ONE AS-OF SNAPSHOT
# ============================================================

def build_asof_snapshot(
    fact: pd.DataFrame,
    abc: pd.DataFrame,
    cutoff_year: int,
    target_year: int,
) -> pd.DataFrame:

    cutoff_population = set(
        fact.loc[
            fact[
                "Year"
            ].eq(
                cutoff_year
            ),
            "Medicine_Key",
        ]
        .dropna()
        .tolist()
    )

    history = (
        fact.loc[
            fact[
                "Year"
            ].le(
                cutoff_year
            )
            &
            fact[
                "Medicine_Key"
            ].isin(
                cutoff_population
            )
        ]
        .copy()
    )

    records = []

    for (
        medicine_key,
        medicine_history,
    ) in history.groupby(
        "Medicine_Key",
        sort=True,
    ):

        medicine_history = (
            medicine_history
            .sort_values(
                "Year"
            )
            .copy()
        )

        last_row = (
            medicine_history.loc[
                medicine_history[
                    "Year"
                ].eq(
                    cutoff_year
                )
            ]
        )

        if len(last_row) != 1:
            raise ValueError(
                f"{medicine_key}: expected "
                f"exactly one record for "
                f"cutoff year {cutoff_year}"
            )

        last_row = (
            last_row.iloc[0]
        )

        issuance = (
            medicine_history[
                "Annual_Issuance"
            ]
            .astype(float)
        )

        positive = (
            issuance.loc[
                issuance.gt(0)
            ]
        )

        history_year_count = int(
            len(
                medicine_history
            )
        )

        positive_years = int(
            len(
                positive
            )
        )

        zero_years = int(
            history_year_count
            - positive_years
        )

        persistence = (
            positive_years
            /
            history_year_count
        )

        mean_issuance = float(
            issuance.mean()
        )

        median_issuance = float(
            issuance.median()
        )

        total_issuance = float(
            issuance.sum()
        )

        mean_positive = (
            float(
                positive.mean()
            )
            if positive_years > 0
            else np.nan
        )

        sd_positive = (
            sample_sd(
                positive
            )
        )

        if (
            positive_years >= 2
            and
            mean_positive > 0
        ):
            cv_positive = (
                sd_positive
                /
                mean_positive
            )
        else:
            cv_positive = np.nan

        years_array = (
            medicine_history[
                "Year"
            ]
            .to_numpy(
                dtype=float
            )
        )

        issuance_array = (
            issuance.to_numpy(
                dtype=float
            )
        )

        historical_trend = (
            trend_slope(
                years_array,
                issuance_array,
            )
        )

        lag1 = float(
            last_row[
                "Annual_Issuance"
            ]
        )

        lag2_row = (
            medicine_history.loc[
                medicine_history[
                    "Year"
                ].eq(
                    cutoff_year - 1
                )
            ]
        )

        if len(
            lag2_row
        ) == 1:
            lag2 = float(
                lag2_row.iloc[0][
                    "Annual_Issuance"
                ]
            )
        else:
            lag2 = np.nan

        if (
            not np.isnan(lag2)
        ):
            moving_average_2 = (
                lag1
                + lag2
            ) / 2
        else:
            moving_average_2 = np.nan

        valid_previous_cost = (
            float(
                last_row[
                    "Average_cost"
                ]
            )
            if bool(
                last_row[
                    "Cost_Valid"
                ]
            )
            else np.nan
        )

        records.append(
            {
                "Medicine_Key":
                    medicine_key,

                "Brand_Name":
                    last_row[
                        "Brand_Name"
                    ],

                "Generic_Name":
                    last_row[
                        "Generic_Name"
                    ],

                "Unit":
                    last_row[
                        "Unit"
                    ],

                "Target_Year":
                    target_year,

                "Feature_Cutoff_Year":
                    cutoff_year,

                "Max_History_Year_Used":
                    int(
                        medicine_history[
                            "Year"
                        ].max()
                    ),

                "History_Years_AsOf":
                    history_year_count,

                "Lag1_Issuance":
                    lag1,

                "Lag2_Issuance":
                    lag2,

                "Moving_Average_2":
                    moving_average_2,

                "Historical_Mean_Issuance":
                    mean_issuance,

                "Historical_Median_Issuance":
                    median_issuance,

                "Historical_Total_Issuance":
                    total_issuance,

                "Positive_Years_AsOf":
                    positive_years,

                "Zero_Years_AsOf":
                    zero_years,

                "Persistence_AsOf":
                    persistence,

                "Mean_Positive_Issuance_AsOf":
                    mean_positive,

                "CV_Positive_Issuance_AsOf":
                    cv_positive,

                "Trend_Slope_AsOf":
                    historical_trend,

                "Lag1_Ending_Balance":
                    last_row[
                        "Ending_Balance"
                    ],

                "Lag1_Ending_Balance_Negative":
                    (
                        bool(
                            last_row[
                                "Ending_Balance"
                            ] < 0
                        )
                        if pd.notna(
                            last_row[
                                "Ending_Balance"
                            ]
                        )
                        else False
                    ),

                "Lag1_Deliveries":
                    last_row[
                        "Qty_in_Deliveries"
                    ],

                "Lag1_Total_In":
                    last_row[
                        "Qty_in_Total_In"
                    ],

                "Lag1_Total_Out":
                    last_row[
                        "Qty_out_Total_Out"
                    ],

                "Lag1_Valid_Average_Cost":
                    valid_previous_cost,

                "Lag1_Cost_Valid":
                    bool(
                        last_row[
                            "Cost_Valid"
                        ]
                    ),

                "Lag1_Data_Quality_Status":
                    last_row[
                        "Data_Quality_Status"
                    ],

                "Feature_Set_Version":
                    FEATURE_SET_VERSION,
            }
        )

    snapshot = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # PRIOR-YEAR ABC
    # --------------------------------------------------------

    prior_abc = (
        abc.loc[
            abc[
                "Year"
            ].eq(
                cutoff_year
            ),
            [
                "Medicine_Key",
                "ABC_Class",
            ],
        ]
        .rename(
            columns={
                "ABC_Class":
                    "ABC_AsOf"
            }
        )
    )

    snapshot = snapshot.merge(
        prior_abc,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # AS-OF BEHAVIOR SEGMENT
    # --------------------------------------------------------

    persistent_mask = (
        snapshot[
            "Persistence_AsOf"
        ]
        .ge(
            PERSISTENCE_THRESHOLD
        )
        &
        snapshot[
            "Positive_Years_AsOf"
        ]
        .gt(0)
    )

    valid_persistent_cv = (
        snapshot.loc[
            persistent_mask
            &
            snapshot[
                "CV_Positive_Issuance_AsOf"
            ].notna(),
            "CV_Positive_Issuance_AsOf",
        ]
    )

    if valid_persistent_cv.empty:
        persistent_cv_threshold = (
            np.nan
        )
    else:
        persistent_cv_threshold = float(
            valid_persistent_cv.median()
        )

    snapshot[
        "Behavior_Segment_AsOf"
    ] = pd.NA

    snapshot.loc[
        snapshot[
            "Positive_Years_AsOf"
        ].eq(0),
        "Behavior_Segment_AsOf",
    ] = "NON_MOVING"

    snapshot.loc[
        snapshot[
            "Positive_Years_AsOf"
        ].gt(0)
        &
        snapshot[
            "Persistence_AsOf"
        ].lt(
            PERSISTENCE_THRESHOLD
        ),
        "Behavior_Segment_AsOf",
    ] = "SPARSE"

    persistent_with_no_cv = (
        persistent_mask
        &
        snapshot[
            "CV_Positive_Issuance_AsOf"
        ]
        .isna()
    )

    snapshot.loc[
        persistent_with_no_cv,
        "Behavior_Segment_AsOf",
    ] = (
        "PERSISTENT_INSUFFICIENT_HISTORY"
    )

    if not np.isnan(
        persistent_cv_threshold
    ):

        snapshot.loc[
            persistent_mask
            &
            snapshot[
                "CV_Positive_Issuance_AsOf"
            ]
            .notna()
            &
            snapshot[
                "CV_Positive_Issuance_AsOf"
            ]
            .le(
                persistent_cv_threshold
            ),
            "Behavior_Segment_AsOf",
        ] = "PERSISTENT_STABLE"

        snapshot.loc[
            persistent_mask
            &
            snapshot[
                "CV_Positive_Issuance_AsOf"
            ]
            .notna()
            &
            snapshot[
                "CV_Positive_Issuance_AsOf"
            ]
            .gt(
                persistent_cv_threshold
            ),
            "Behavior_Segment_AsOf",
        ] = "PERSISTENT_VARIABLE"

    snapshot[
        "Persistent_CV_Threshold_AsOf"
    ] = persistent_cv_threshold

    return snapshot


# ============================================================
# BUILD SUPERVISED DATASET
# ============================================================

def build_supervised_dataset(
    fact: pd.DataFrame,
    abc: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

    for target_year in (
        SUPERVISED_TARGET_YEARS
    ):

        cutoff_year = (
            target_year - 1
        )

        features = (
            build_asof_snapshot(
                fact=fact,
                abc=abc,
                cutoff_year=cutoff_year,
                target_year=target_year,
            )
        )

        target = (
            fact.loc[
                fact[
                    "Year"
                ].eq(
                    target_year
                ),
                [
                    "Medicine_Key",
                    "Annual_Issuance",
                ],
            ]
            .rename(
                columns={
                    "Annual_Issuance":
                        "Target_Annual_Issuance"
                }
            )
        )

        dataset = features.merge(
            target,
            on="Medicine_Key",
            how="inner",
            validate="one_to_one",
        )

        dataset[
            "Target_Occurrence"
        ] = (
            dataset[
                "Target_Annual_Issuance"
            ]
            .gt(0)
        )

        dataset[
            "Target_Log1p_Issuance"
        ] = np.log1p(
            dataset[
                "Target_Annual_Issuance"
            ]
        )

        frames.append(
            dataset
        )

    supervised = pd.concat(
        frames,
        ignore_index=True,
    )

    return supervised


# ============================================================
# BUILD NEXT-YEAR SCORING DATASET
# ============================================================

def build_scoring_dataset(
    fact: pd.DataFrame,
    abc: pd.DataFrame,
) -> pd.DataFrame:

    cutoff_year = (
        SCORING_TARGET_YEAR - 1
    )

    scoring = (
        build_asof_snapshot(
            fact=fact,
            abc=abc,
            cutoff_year=cutoff_year,
            target_year=SCORING_TARGET_YEAR,
        )
    )

    return scoring


# ============================================================
# LEAKAGE AUDIT
# ============================================================

def validate_no_leakage(
    dataset: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:

    audit_rows = []

    for (
        target_year,
        year_df,
    ) in dataset.groupby(
        "Target_Year",
        sort=True,
    ):

        violations = (
            year_df[
                "Max_History_Year_Used"
            ]
            .ge(
                target_year
            )
        )

        cutoff_mismatch = (
            year_df[
                "Feature_Cutoff_Year"
            ]
            .ne(
                target_year - 1
            )
        )

        rows_with_future_history = int(
            violations.sum()
        )

        rows_with_cutoff_error = int(
            cutoff_mismatch.sum()
        )

        audit_pass = (
            rows_with_future_history == 0
            and
            rows_with_cutoff_error == 0
        )

        audit_rows.append(
            {
                "Dataset":
                    dataset_name,

                "Target_Year":
                    int(
                        target_year
                    ),

                "Rows":
                    len(year_df),

                "Feature_Cutoff_Year_Min":
                    int(
                        year_df[
                            "Feature_Cutoff_Year"
                        ].min()
                    ),

                "Feature_Cutoff_Year_Max":
                    int(
                        year_df[
                            "Feature_Cutoff_Year"
                        ].max()
                    ),

                "Max_History_Year_Used":
                    int(
                        year_df[
                            "Max_History_Year_Used"
                        ].max()
                    ),

                "Future_History_Violations":
                    rows_with_future_history,

                "Cutoff_Errors":
                    rows_with_cutoff_error,

                "Leakage_Audit_Pass":
                    audit_pass,
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    if not (
        audit[
            "Leakage_Audit_Pass"
        ]
        .all()
    ):
        raise ValueError(
            "LEAKAGE AUDIT FAILED. "
            "Forecast dataset build stopped."
        )

    return audit


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_structure(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
    fact: pd.DataFrame,
) -> None:

    duplicate_supervised = int(
        supervised.duplicated(
            subset=[
                "Medicine_Key",
                "Target_Year",
            ]
        ).sum()
    )

    if duplicate_supervised:
        raise ValueError(
            "Duplicate supervised "
            "Medicine_Key-Target_Year rows."
        )

    duplicate_scoring = int(
        scoring[
            "Medicine_Key"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_scoring:
        raise ValueError(
            "Duplicate scoring Medicine_Key."
        )

    if supervised[
        "Target_Annual_Issuance"
    ].isna().any():

        raise ValueError(
            "Missing supervised target found."
        )

    if (
        supervised[
            "Target_Annual_Issuance"
        ]
        .lt(0)
        .any()
    ):
        raise ValueError(
            "Negative target issuance found."
        )

    # --------------------------------------------------------
    # OBSERVED POPULATION CONTRACT
    # --------------------------------------------------------

    for target_year in (
        SUPERVISED_TARGET_YEARS
    ):

        prior_keys = set(
            fact.loc[
                fact[
                    "Year"
                ].eq(
                    target_year - 1
                ),
                "Medicine_Key",
            ]
        )

        target_keys = set(
            fact.loc[
                fact[
                    "Year"
                ].eq(
                    target_year
                ),
                "Medicine_Key",
            ]
        )

        expected_keys = (
            prior_keys
            &
            target_keys
        )

        actual_keys = set(
            supervised.loc[
                supervised[
                    "Target_Year"
                ].eq(
                    target_year
                ),
                "Medicine_Key",
            ]
        )

        if (
            actual_keys
            != expected_keys
        ):
            raise ValueError(
                "Supervised population mismatch "
                f"for target year {target_year}. "
                f"Expected {len(expected_keys):,} "
                f"observed t-1/t identities; "
                f"found {len(actual_keys):,}."
            )

    latest_completed_keys = set(
        fact.loc[
            fact[
                "Year"
            ].eq(
                RUNTIME_CONTEXT.latest_completed_year
            ),
            "Medicine_Key",
        ]
    )

    scoring_keys = set(
        scoring[
            "Medicine_Key"
        ]
    )

    if (
        scoring_keys
        != latest_completed_keys
    ):
        raise ValueError(
            "Scoring population must equal "
            "the medicines actually observed "
            "in the latest completed year."
        )

    if not (
        scoring[
            "Target_Year"
        ]
        .eq(
            SCORING_TARGET_YEAR
        )
        .all()
    ):
        raise ValueError(
            "Scoring target year validation failed."
        )

    if not (
        scoring[
            "Feature_Cutoff_Year"
        ]
        .eq(
            RUNTIME_CONTEXT.latest_completed_year
        )
        .all()
    ):
        raise ValueError(
            "Scoring feature-cutoff validation failed."
        )

    # --------------------------------------------------------
    # FROZEN BASELINE REGRESSION CONTRACT
    # --------------------------------------------------------

    if RUNTIME_CONTEXT.is_baseline:

        if SUPERVISED_TARGET_YEARS != [
            2022,
            2023,
            2024,
            2025,
        ]:
            raise ValueError(
                "Frozen baseline supervised-year "
                "configuration changed."
            )

        if SCORING_TARGET_YEAR != 2026:
            raise ValueError(
                "Frozen baseline scoring year changed."
            )

        if len(supervised) != 10932:
            raise ValueError(
                "Frozen baseline expected "
                "10,932 supervised rows; "
                f"found {len(supervised):,}."
            )

        if len(scoring) != 2733:
            raise ValueError(
                "Frozen baseline expected "
                "2,733 scoring rows; "
                f"found {len(scoring):,}."
            )


# ============================================================
# BACKTEST FOLDS
# ============================================================

def build_backtest_folds(
    supervised: pd.DataFrame,
) -> pd.DataFrame:

    first_supervised_year = (
        SUPERVISED_TARGET_YEARS[0]
    )

    first_test_year = (
        first_supervised_year + 2
    )

    test_years = [
        year
        for year
        in SUPERVISED_TARGET_YEARS
        if year >= first_test_year
    ]

    rows = []

    for test_year in test_years:

        spec = {
            "Fold_Name":
                f"BACKTEST_{test_year}",

            "Train_Target_Year_Start":
                first_supervised_year,

            "Train_Target_Year_End":
                test_year - 1,

            "Test_Target_Year":
                test_year,
        }

        train_mask = (
            supervised[
                "Target_Year"
            ]
            .between(
                spec[
                    "Train_Target_Year_Start"
                ],
                spec[
                    "Train_Target_Year_End"
                ],
            )
        )

        test_mask = (
            supervised[
                "Target_Year"
            ]
            .eq(
                spec[
                    "Test_Target_Year"
                ]
            )
        )

        if not train_mask.any():
            raise ValueError(
                f"{spec['Fold_Name']}: "
                "no chronological training rows."
            )

        if not test_mask.any():
            raise ValueError(
                f"{spec['Fold_Name']}: "
                "no chronological test rows."
            )

        rows.append(
            {
                **spec,

                "Train_Row_Count":
                    int(
                        train_mask.sum()
                    ),

                "Test_Row_Count":
                    int(
                        test_mask.sum()
                    ),

                "Random_Split_Used":
                    False,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FEATURE DICTIONARY
# ============================================================

def build_feature_dictionary() -> pd.DataFrame:

    rows = [
        {
            "Field":
                "Medicine_Key",
            "Role":
                "IDENTIFIER",
            "Model_Feature":
                False,
            "Timing_Rule":
                "Stable identifier",
            "Notes":
                "Not used as a predictive numeric feature.",
        },
        {
            "Field":
                "Target_Year",
            "Role":
                "FORECAST_ORIGIN_METADATA",
            "Model_Feature":
                False,
            "Timing_Rule":
                "Known before forecast",
            "Notes":
                "Defines the annual period being predicted.",
        },
        {
            "Field":
                "Lag1_Issuance",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Target year minus 1",
            "Notes":
                "Previous completed year's recorded issuance.",
        },
        {
            "Field":
                "Lag2_Issuance",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Target year minus 2",
            "Notes":
                "Second previous completed year's issuance.",
        },
        {
            "Field":
                "Moving_Average_2",
            "Role":
                "PREDICTOR / BASELINE",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Uses lag 1 and lag 2 only",
            "Notes":
                "Also used as two-year moving-average benchmark.",
        },
        {
            "Field":
                "Historical_Mean_Issuance",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "History through cutoff year only",
            "Notes":
                "No target-year observations included.",
        },
        {
            "Field":
                "Persistence_AsOf",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "History through cutoff year only",
            "Notes":
                "Positive issuance years divided by available historical years.",
        },
        {
            "Field":
                "CV_Positive_Issuance_AsOf",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Positive historical issuance through cutoff only",
            "Notes":
                "Missing when fewer than two positive observations exist.",
        },
        {
            "Field":
                "Trend_Slope_AsOf",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "History through cutoff year only",
            "Notes":
                "Simple annual issuance trend slope.",
        },
        {
            "Field":
                "Lag1_Ending_Balance",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Previous completed year",
            "Notes":
                "Missing values remain missing; not converted to zero.",
        },
        {
            "Field":
                "Lag1_Deliveries",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Previous completed year",
            "Notes":
                "No target-year deliveries are used.",
        },
        {
            "Field":
                "Lag1_Valid_Average_Cost",
            "Role":
                "PREDICTOR",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Previous completed year",
            "Notes":
                "Invalid prior cost is represented as missing.",
        },
        {
            "Field":
                "ABC_AsOf",
            "Role":
                "PREDICTOR / SEGMENT",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Previous completed year's ABC",
            "Notes":
                "Economic classification only; no clinical interpretation.",
        },
        {
            "Field":
                "Behavior_Segment_AsOf",
            "Role":
                "SEGMENT",
            "Model_Feature":
                True,
            "Timing_Rule":
                "Calculated only from history through cutoff",
            "Notes":
                f"Not the full {HISTORICAL_YEARS[0]}-"
                f"{HISTORICAL_YEARS[-1]} descriptive segment.",
        },
        {
            "Field":
                "Target_Annual_Issuance",
            "Role":
                "TARGET",
            "Model_Feature":
                False,
            "Timing_Rule":
                "Target year",
            "Notes":
                "Prediction outcome; never enters predictor matrix.",
        },
        {
            "Field":
                "Target_Occurrence",
            "Role":
                "CLASSIFICATION_TARGET",
            "Model_Feature":
                False,
            "Timing_Rule":
                "Derived from target year",
            "Notes":
                "Used by the hurdle occurrence classifier.",
        },
        {
            "Field":
                "Target_Log1p_Issuance",
            "Role":
                "REGRESSION_TARGET",
            "Model_Feature":
                False,
            "Timing_Rule":
                "Derived from target year",
            "Notes":
                "Optional transformed regression target.",
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# WRITE DATABASE
# ============================================================

def write_database(
    supervised: pd.DataFrame,
    scoring: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    folds: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(
            DATABASE_PATH
        )
    )

    try:

        connection.register(
            "supervised_df",
            supervised,
        )

        connection.register(
            "scoring_df",
            scoring,
        )

        connection.register(
            "leakage_df",
            leakage_audit,
        )

        connection.register(
            "folds_df",
            folds,
        )

        connection.register(
            "dictionary_df",
            dictionary,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecasting_supervised AS
            SELECT *
            FROM supervised_df
            """
        )

        scoring_table = (
            f"forecasting_scoring_"
            f"{SCORING_TARGET_YEAR}"
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {scoring_table} AS
            SELECT *
            FROM scoring_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_leakage_audit AS
            SELECT *
            FROM leakage_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_backtest_folds AS
            SELECT *
            FROM folds_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_feature_dictionary AS
            SELECT *
            FROM dictionary_df
            """
        )

    finally:
        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 96)
    print("CDH INVENTORY DSS")
    print("LEAKAGE-SAFE ANNUAL FORECASTING DATASET")
    print("=" * 96)
    print()

    fact = load_fact()
    abc = load_abc()

    print(
        "Historical medicine-year rows:",
        f"{len(fact):,}",
    )

    supervised = (
        build_supervised_dataset(
            fact,
            abc,
        )
    )

    scoring = (
        build_scoring_dataset(
            fact,
            abc,
        )
    )

    validate_structure(
        supervised,
        scoring,
        fact,
    )

    print(
        "Structural validation: PASS"
    )

    supervised_audit = (
        validate_no_leakage(
            supervised,
            (
                f"SUPERVISED_"
                f"{SUPERVISED_TARGET_YEARS[0]}_"
                f"{SUPERVISED_TARGET_YEARS[-1]}"
            ),
        )
    )

    scoring_audit = (
        validate_no_leakage(
            scoring,
            f"SCORING_{SCORING_TARGET_YEAR}",
        )
    )

    leakage_audit = pd.concat(
        [
            supervised_audit,
            scoring_audit,
        ],
        ignore_index=True,
    )

    print(
        "Temporal leakage audit: PASS"
    )

    folds = (
        build_backtest_folds(
            supervised
        )
    )

    dictionary = (
        build_feature_dictionary()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    supervised.to_parquet(
        OUTPUT_DIR
        / (
            f"forecasting_supervised_"
            f"{SUPERVISED_TARGET_YEARS[0]}_"
            f"{SUPERVISED_TARGET_YEARS[-1]}.parquet"
        ),
        index=False,
        engine="pyarrow",
    )

    supervised.to_csv(
        OUTPUT_DIR
        / (
            f"forecasting_supervised_"
            f"{SUPERVISED_TARGET_YEARS[0]}_"
            f"{SUPERVISED_TARGET_YEARS[-1]}.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    scoring.to_parquet(
        OUTPUT_DIR
        / (
            f"forecasting_scoring_"
            f"{SCORING_TARGET_YEAR}.parquet"
        ),
        index=False,
        engine="pyarrow",
    )

    leakage_audit.to_csv(
        OUTPUT_DIR
        / "forecast_leakage_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    folds.to_csv(
        OUTPUT_DIR
        / "forecast_backtest_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dictionary.to_csv(
        OUTPUT_DIR
        / "forecast_feature_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        supervised,
        scoring,
        leakage_audit,
        folds,
        dictionary,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 96)
    print("SUPERVISED DATASET")
    print("-" * 96)

    supervised_summary = (
        supervised.groupby(
            "Target_Year"
        )
        .agg(
            Rows=(
                "Medicine_Key",
                "size",
            ),
            Positive_Targets=(
                "Target_Occurrence",
                "sum",
            ),
            Total_Target_Issuance=(
                "Target_Annual_Issuance",
                "sum",
            ),
            Mean_History_Years=(
                "History_Years_AsOf",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        supervised_summary
        .to_string(
            index=False
        )
    )

    print()
    print("-" * 96)
    print("LEAKAGE AUDIT")
    print("-" * 96)

    print(
        leakage_audit
        .to_string(
            index=False
        )
    )

    print()
    print("-" * 96)
    print("BACKTEST FOLDS")
    print("-" * 96)

    print(
        folds
        .to_string(
            index=False
        )
    )

    print()
    print("-" * 96)
    print(
        f"{SCORING_TARGET_YEAR} SCORING FEATURES"
    )
    print("-" * 96)

    print(
        "Rows:",
        f"{len(scoring):,}",
    )

    print(
        "Feature cutoff year:",
        int(
            scoring[
                "Feature_Cutoff_Year"
            ]
            .unique()[0]
        ),
    )

    print(
        f"No {SCORING_TARGET_YEAR} observed issuance "
        "has been included as a feature."
    )

    print()
    print("-" * 96)
    print("IMPORTANT GOVERNANCE RULE")
    print("-" * 96)

    print(
        "The full-history table "
        f"'demand_behavior_{HISTORICAL_YEARS[0]}_"
        f"{HISTORICAL_YEARS[-1]}' "
        "is DESCRIPTIVE ONLY."
    )

    print(
        "Forecast models must use "
        "'Behavior_Segment_AsOf' and "
        "other as-of features from this "
        "forecasting dataset."
    )

    print(
        "No random train/test split "
        "is permitted."
    )

    print()
    print("=" * 96)
    print(
        "FORECASTING DATASET BUILD COMPLETE"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
    