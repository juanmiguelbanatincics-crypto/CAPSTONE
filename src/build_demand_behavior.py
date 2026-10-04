from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


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
    / "demand_behavior"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PARAMETERS
# ============================================================

EXPECTED_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

PERSISTENT_THRESHOLD = 0.80


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
                ABC_Class,
                ABC_Eligibility
            FROM abc_classification_annual
            ORDER BY Medicine_Key, Year
            """
        ).df()

    finally:
        connection.close()

    return dataframe


# ============================================================
# BEHAVIOR FEATURES
# ============================================================

def population_sd(
    values: pd.Series,
) -> float | np.nan:

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


def calculate_behavior_features(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for (
        medicine_key,
        medicine_df,
    ) in fact.groupby(
        "Medicine_Key",
        sort=True,
    ):

        medicine_df = (
            medicine_df
            .sort_values("Year")
            .copy()
        )

        years_observed = int(
            medicine_df[
                "Year"
            ].nunique()
        )

        positive = (
            medicine_df.loc[
                medicine_df[
                    "Annual_Issuance"
                ].gt(0),
                "Annual_Issuance",
            ]
            .astype(float)
        )

        positive_years = int(
            len(positive)
        )

        zero_years = int(
            years_observed
            - positive_years
        )

        persistence = (
            positive_years
            / years_observed
            if years_observed > 0
            else np.nan
        )

        total_issuance = float(
            medicine_df[
                "Annual_Issuance"
            ]
            .sum()
        )

        mean_all_years = float(
            medicine_df[
                "Annual_Issuance"
            ]
            .mean()
        )

        median_all_years = float(
            medicine_df[
                "Annual_Issuance"
            ]
            .median()
        )

        mean_positive = (
            float(
                positive.mean()
            )
            if positive_years > 0
            else np.nan
        )

        median_positive = (
            float(
                positive.median()
            )
            if positive_years > 0
            else np.nan
        )

        sd_positive = (
            population_sd(
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
                / mean_positive
            )
        else:
            cv_positive = np.nan

        # ----------------------------------------------------
        # SIMPLE TREND SLOPE
        # ----------------------------------------------------

        years = (
            medicine_df[
                "Year"
            ]
            .to_numpy(
                dtype=float
            )
        )

        issuance = (
            medicine_df[
                "Annual_Issuance"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(years) >= 2:
            trend_slope = float(
                np.polyfit(
                    years,
                    issuance,
                    1,
                )[0]
            )
        else:
            trend_slope = np.nan

        first_year_issuance = float(
            medicine_df.iloc[0][
                "Annual_Issuance"
            ]
        )

        latest_year_issuance = float(
            medicine_df.iloc[-1][
                "Annual_Issuance"
            ]
        )

        # ----------------------------------------------------
        # PRELIMINARY MAIN BEHAVIOR GROUP
        # ----------------------------------------------------

        if positive_years == 0:
            preliminary_group = (
                "NON_MOVING"
            )

        elif (
            persistence
            >= PERSISTENT_THRESHOLD
        ):
            preliminary_group = (
                "PERSISTENT"
            )

        else:
            preliminary_group = (
                "SPARSE"
            )

        first_row = (
            medicine_df.iloc[0]
        )

        records.append(
            {
                "Medicine_Key":
                    medicine_key,

                "Brand_Name":
                    first_row[
                        "Brand_Name"
                    ],

                "Generic_Name":
                    first_row[
                        "Generic_Name"
                    ],

                "Unit":
                    first_row[
                        "Unit"
                    ],

                "Years_Observed":
                    years_observed,

                "Positive_Issuance_Years":
                    positive_years,

                "Zero_Issuance_Years":
                    zero_years,

                "Persistence":
                    persistence,

                "Total_Issuance_2021_2025":
                    total_issuance,

                "Mean_Annual_Issuance":
                    mean_all_years,

                "Median_Annual_Issuance":
                    median_all_years,

                "Mean_Positive_Issuance":
                    mean_positive,

                "Median_Positive_Issuance":
                    median_positive,

                "SD_Positive_Issuance":
                    sd_positive,

                "CV_Positive_Issuance":
                    cv_positive,

                "Trend_Slope":
                    trend_slope,

                "Issuance_2021":
                    first_year_issuance,

                "Issuance_2025":
                    latest_year_issuance,

                "Preliminary_Behavior":
                    preliminary_group,
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# FINAL SEGMENTATION
# ============================================================

def assign_final_segments(
    behavior: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    float,
]:

    result = behavior.copy()

    persistent_mask = (
        result[
            "Preliminary_Behavior"
        ]
        .eq("PERSISTENT")
    )

    eligible_persistent_cv = (
        result.loc[
            persistent_mask
            &
            result[
                "CV_Positive_Issuance"
            ].notna(),
            "CV_Positive_Issuance",
        ]
    )

    if eligible_persistent_cv.empty:
        raise ValueError(
            "No persistent medicines have "
            "a valid CV."
        )

    median_cv = float(
        eligible_persistent_cv.median()
    )

    result[
        "Behavior_Segment"
    ] = pd.NA

    # Non-moving
    result.loc[
        result[
            "Preliminary_Behavior"
        ].eq("NON_MOVING"),
        "Behavior_Segment",
    ] = "NON_MOVING"

    # Sparse
    result.loc[
        result[
            "Preliminary_Behavior"
        ].eq("SPARSE"),
        "Behavior_Segment",
    ] = "SPARSE"

    # Persistent stable
    result.loc[
        persistent_mask
        &
        result[
            "CV_Positive_Issuance"
        ].le(median_cv),
        "Behavior_Segment",
    ] = "PERSISTENT_STABLE"

    # Persistent variable
    result.loc[
        persistent_mask
        &
        result[
            "CV_Positive_Issuance"
        ].gt(median_cv),
        "Behavior_Segment",
    ] = "PERSISTENT_VARIABLE"

    result[
        "Persistent_CV_Median_Threshold"
    ] = median_cv

    result[
        "Behavior_Method"
    ] = (
        "PERSISTENCE_PLUS_MEDIAN_POSITIVE_CV"
    )

    return (
        result,
        median_cv,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_behavior(
    behavior: pd.DataFrame,
) -> None:

    if len(
        behavior
    ) != 2733:

        raise ValueError(
            f"Expected 2,733 medicines, "
            f"found {len(behavior):,}"
        )

    if (
        behavior[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate Medicine_Key "
            "found in behavior table."
        )

    if not (
        behavior[
            "Years_Observed"
        ]
        .eq(5)
        .all()
    ):

        raise ValueError(
            "Every medicine must have "
            "five annual observations."
        )

    allowed_segments = {
        "PERSISTENT_STABLE",
        "PERSISTENT_VARIABLE",
        "SPARSE",
        "NON_MOVING",
    }

    actual_segments = set(
        behavior[
            "Behavior_Segment"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if actual_segments != allowed_segments:
        print(
            "Observed behavior segments:",
            actual_segments,
        )

        missing = (
            allowed_segments
            - actual_segments
        )

        unexpected = (
            actual_segments
            - allowed_segments
        )

        if unexpected:
            raise ValueError(
                f"Unexpected segments: "
                f"{unexpected}"
            )

        if missing:
            print(
                "Warning: no medicines "
                "occurred in segment(s):",
                missing,
            )

    missing_segment = int(
        behavior[
            "Behavior_Segment"
        ]
        .isna()
        .sum()
    )

    if missing_segment:
        raise ValueError(
            f"{missing_segment} medicines "
            "lack a final behavior segment."
        )

    # Logical rule validation
    non_moving_error = int(
        (
            behavior[
                "Behavior_Segment"
            ]
            .eq("NON_MOVING")
            &
            behavior[
                "Positive_Issuance_Years"
            ]
            .ne(0)
        )
        .sum()
    )

    if non_moving_error:
        raise ValueError(
            "NON_MOVING logical validation "
            "failed."
        )

    sparse_error = int(
        (
            behavior[
                "Behavior_Segment"
            ]
            .eq("SPARSE")
            &
            ~behavior[
                "Positive_Issuance_Years"
            ]
            .isin(
                [1, 2, 3]
            )
        )
        .sum()
    )

    if sparse_error:
        raise ValueError(
            "SPARSE logical validation failed."
        )

    persistent_error = int(
        (
            behavior[
                "Behavior_Segment"
            ]
            .isin(
                [
                    "PERSISTENT_STABLE",
                    "PERSISTENT_VARIABLE",
                ]
            )
            &
            behavior[
                "Positive_Issuance_Years"
            ]
            .lt(4)
        )
        .sum()
    )

    if persistent_error:
        raise ValueError(
            "PERSISTENT logical validation "
            "failed."
        )


# ============================================================
# BEHAVIOR SUMMARY
# ============================================================

def build_behavior_summary(
    behavior: pd.DataFrame,
) -> pd.DataFrame:

    order = [
        "PERSISTENT_STABLE",
        "PERSISTENT_VARIABLE",
        "SPARSE",
        "NON_MOVING",
    ]

    rows = []

    total_medicines = len(
        behavior
    )

    total_issuance = float(
        behavior[
            "Total_Issuance_2021_2025"
        ]
        .sum()
    )

    for segment in order:

        segment_df = (
            behavior.loc[
                behavior[
                    "Behavior_Segment"
                ]
                .eq(segment)
            ]
        )

        segment_issuance = float(
            segment_df[
                "Total_Issuance_2021_2025"
            ]
            .sum()
        )

        rows.append(
            {
                "Behavior_Segment":
                    segment,

                "Medicine_Count":
                    len(segment_df),

                "Medicine_Share_Pct":
                    round(
                        len(segment_df)
                        /
                        total_medicines
                        * 100,
                        4,
                    ),

                "Total_Issuance_2021_2025":
                    segment_issuance,

                "Issuance_Share_Pct":
                    (
                        round(
                            segment_issuance
                            /
                            total_issuance
                            * 100,
                            4,
                        )
                        if total_issuance > 0
                        else np.nan
                    ),

                "Median_Persistence":
                    (
                        float(
                            segment_df[
                                "Persistence"
                            ]
                            .median()
                        )
                        if len(
                            segment_df
                        ) > 0
                        else np.nan
                    ),

                "Median_Positive_CV":
                    (
                        float(
                            segment_df[
                                "CV_Positive_Issuance"
                            ]
                            .median()
                        )
                        if segment_df[
                            "CV_Positive_Issuance"
                        ]
                        .notna()
                        .any()
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# JOIN WITH ANNUAL ABC
# ============================================================

def build_abc_behavior_panel(
    behavior: pd.DataFrame,
    abc: pd.DataFrame,
) -> pd.DataFrame:

    joined = abc.merge(
        behavior[
            [
                "Medicine_Key",
                "Behavior_Segment",
                "Persistence",
                "Positive_Issuance_Years",
                "CV_Positive_Issuance",
                "Persistent_CV_Median_Threshold",
            ]
        ],
        on="Medicine_Key",
        how="left",
        validate="many_to_one",
    )

    joined[
        "ABC_Behavior_Segment"
    ] = (
        joined[
            "ABC_Class"
        ]
        .astype("string")
        +
        "__"
        +
        joined[
            "Behavior_Segment"
        ]
        .astype("string")
    )

    return joined


# ============================================================
# ABC × BEHAVIOR MATRIX
# ============================================================

def build_matrix(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    abc_order = [
        "A",
        "B",
        "C",
        "UNCLASSIFIED_COST",
    ]

    behavior_order = [
        "PERSISTENT_STABLE",
        "PERSISTENT_VARIABLE",
        "SPARSE",
        "NON_MOVING",
    ]

    for year in sorted(
        panel[
            "Year"
        ]
        .unique()
        .tolist()
    ):

        year_df = panel.loc[
            panel[
                "Year"
            ].eq(year)
        ]

        for abc_class in abc_order:

            for behavior_segment in (
                behavior_order
            ):

                subset = (
                    year_df.loc[
                        year_df[
                            "ABC_Class"
                        ]
                        .eq(
                            abc_class
                        )
                        &
                        year_df[
                            "Behavior_Segment"
                        ]
                        .eq(
                            behavior_segment
                        )
                    ]
                )

                rows.append(
                    {
                        "Year":
                            int(year),

                        "ABC_Class":
                            abc_class,

                        "Behavior_Segment":
                            behavior_segment,

                        "Medicine_Count":
                            len(subset),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LATEST-YEAR MATRIX
# ============================================================

def build_latest_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:

    latest_year = int(
        matrix[
            "Year"
        ]
        .max()
    )

    latest = (
        matrix.loc[
            matrix[
                "Year"
            ]
            .eq(
                latest_year
            )
        ]
        .pivot(
            index="ABC_Class",
            columns="Behavior_Segment",
            values="Medicine_Count",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    latest[
        "Year"
    ] = latest_year

    return latest


# ============================================================
# WRITE DUCKDB
# ============================================================

def write_database(
    behavior: pd.DataFrame,
    summary: pd.DataFrame,
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    latest_matrix: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "behavior_df",
            behavior,
        )

        connection.register(
            "summary_df",
            summary,
        )

        connection.register(
            "panel_df",
            panel,
        )

        connection.register(
            "matrix_df",
            matrix,
        )

        connection.register(
            "latest_matrix_df",
            latest_matrix,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            demand_behavior_2021_2025 AS
            SELECT *
            FROM behavior_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            demand_behavior_summary AS
            SELECT *
            FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            abc_behavior_annual AS
            SELECT *
            FROM panel_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            abc_behavior_matrix_annual AS
            SELECT *
            FROM matrix_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            abc_behavior_matrix_latest AS
            SELECT *
            FROM latest_matrix_df
            """
        )

    finally:
        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 92)
    print("CDH INVENTORY DSS")
    print("2021-2025 DEMAND-BEHAVIOR SEGMENTATION")
    print("=" * 92)
    print()

    fact = load_fact()

    abc = load_abc()

    print(
        "Medicine-year rows loaded:",
        f"{len(fact):,}",
    )

    print(
        "Unique medicines:",
        f"{fact['Medicine_Key'].nunique():,}",
    )

    behavior_features = (
        calculate_behavior_features(
            fact
        )
    )

    behavior, median_cv = (
        assign_final_segments(
            behavior_features
        )
    )

    validate_behavior(
        behavior
    )

    print(
        "Behavior structural validation: PASS"
    )

    print()
    print(
        "Persistent-medicine median "
        "positive-demand CV threshold:",
        f"{median_cv:.6f}",
    )

    summary = build_behavior_summary(
        behavior
    )

    panel = build_abc_behavior_panel(
        behavior,
        abc,
    )

    matrix = build_matrix(
        panel
    )

    latest_matrix = (
        build_latest_matrix(
            matrix
        )
    )

    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    behavior.to_parquet(
        OUTPUT_DIR
        / "demand_behavior_2021_2025.parquet",
        index=False,
        engine="pyarrow",
    )

    behavior.to_csv(
        OUTPUT_DIR
        / "demand_behavior_2021_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "demand_behavior_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    matrix.to_csv(
        OUTPUT_DIR
        / "abc_behavior_matrix_annual.csv",
        index=False,
        encoding="utf-8-sig",
    )

    latest_matrix.to_csv(
        OUTPUT_DIR
        / "abc_behavior_matrix_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        behavior,
        summary,
        panel,
        matrix,
        latest_matrix,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 92)
    print("DEMAND-BEHAVIOR SUMMARY")
    print("-" * 92)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("-" * 92)
    print("LATEST COMPLETED YEAR ABC × BEHAVIOR MATRIX")
    print("-" * 92)

    print(
        latest_matrix.to_string(
            index=False
        )
    )

    print()
    print("-" * 92)
    print("METHOD RULES")
    print("-" * 92)

    print(
        "Persistence = positive issuance "
        "years / 5 completed annual years."
    )

    print(
        "Persistent: persistence >= 0.80 "
        "(at least 4 of 5 years)."
    )

    print(
        "Sparse: issuance in 1-3 of 5 years."
    )

    print(
        "Non-Moving: zero recorded issuance "
        "in all five years."
    )

    print(
        "Persistent medicines are divided "
        "into Stable vs Variable using the "
        "median CV of positive annual issuance."
    )

    print(
        "ABC remains an economic classification; "
        "behavior describes issuance history."
    )

    print()
    print("=" * 92)
    print(
        "DEMAND-BEHAVIOR SEGMENTATION COMPLETE"
    )
    print("=" * 92)

    print(
        "No validated historical record "
        "was modified."
    )


if __name__ == "__main__":
    main()