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
    / "abc"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ABC PARAMETERS
# ============================================================

A_THRESHOLD = 0.70
B_THRESHOLD = 0.90


# ============================================================
# LOAD VALIDATED FACT
# ============================================================

def load_fact() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        fact = connection.execute(
            """
            SELECT
                Medicine_Key,
                Year,
                Brand_Name,
                Generic_Name,
                Unit,
                Annual_Issuance,
                Average_cost,
                Cost_Valid,
                Annual_Consumption_Value,
                ACV_Computable,
                ACV_Status,
                Data_Quality_Status,
                Source_File
            FROM fact_inventory_annual
            ORDER BY Year, Medicine_Key
            """
        ).df()

    finally:
        connection.close()

    return fact


# ============================================================
# CLASSIFY ONE YEAR
# ============================================================

def classify_year(
    year_df: pd.DataFrame,
) -> pd.DataFrame:

    result = year_df.copy()

    result[
        "ABC_Class"
    ] = pd.NA

    result[
        "ABC_Eligibility"
    ] = np.where(
        result[
            "ACV_Computable"
        ],
        "ELIGIBLE",
        "UNCLASSIFIED_COST",
    )

    # --------------------------------------------------------
    # ELIGIBLE ACV RECORDS
    # --------------------------------------------------------

    eligible = (
        result.loc[
            result[
                "ACV_Computable"
            ]
        ]
        .copy()
    )

    if eligible.empty:
        raise ValueError(
            f"No ABC-eligible records "
            f"for year {year_df['Year'].iloc[0]}"
        )

    # Stable deterministic ordering:
    # 1. higher ACV first
    # 2. Medicine_Key breaks ties
    eligible = eligible.sort_values(
        by=[
            "Annual_Consumption_Value",
            "Medicine_Key",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    total_known_acv = float(
        eligible[
            "Annual_Consumption_Value"
        ].sum()
    )

    eligible[
        "Known_ACV_Total_Year"
    ] = total_known_acv

    # --------------------------------------------------------
    # HANDLE ZERO TOTAL VALUE
    # --------------------------------------------------------

    if total_known_acv <= 0:

        eligible[
            "ACV_Share"
        ] = 0.0

        eligible[
            "Cumulative_ACV"
        ] = 0.0

        eligible[
            "Cumulative_ACV_Share"
        ] = 0.0

        eligible[
            "Cumulative_Share_Before"
        ] = 0.0

        eligible[
            "ABC_Class"
        ] = "C"

    else:

        eligible[
            "ACV_Share"
        ] = (
            eligible[
                "Annual_Consumption_Value"
            ]
            /
            total_known_acv
        )

        eligible[
            "Cumulative_ACV"
        ] = (
            eligible[
                "Annual_Consumption_Value"
            ]
            .cumsum()
        )

        eligible[
            "Cumulative_ACV_Share"
        ] = (
            eligible[
                "Cumulative_ACV"
            ]
            /
            total_known_acv
        )

        eligible[
            "Cumulative_Share_Before"
        ] = (
            eligible[
                "Cumulative_ACV_Share"
            ]
            -
            eligible[
                "ACV_Share"
            ]
        )

        # Crossing-item rule:
        #
        # If the cumulative share BEFORE
        # the item is still below 70%,
        # the item belongs to A.
        #
        # Likewise for the 90% B boundary.
        eligible[
            "ABC_Class"
        ] = np.select(
            [
                eligible[
                    "Cumulative_Share_Before"
                ].lt(
                    A_THRESHOLD
                ),

                eligible[
                    "Cumulative_Share_Before"
                ].lt(
                    B_THRESHOLD
                ),
            ],
            [
                "A",
                "B",
            ],
            default="C",
        )

    # --------------------------------------------------------
    # MAP ELIGIBLE RESULTS BACK
    # --------------------------------------------------------

    classification_columns = [
        "Medicine_Key",
        "ABC_Class",
        "ABC_Eligibility",
        "Known_ACV_Total_Year",
        "ACV_Share",
        "Cumulative_ACV",
        "Cumulative_ACV_Share",
        "Cumulative_Share_Before",
    ]

    result = result.drop(
        columns=[
            "ABC_Class",
            "ABC_Eligibility",
        ]
    ).merge(
        eligible[
            classification_columns
        ],
        on="Medicine_Key",
        how="left",
    )

    # --------------------------------------------------------
    # INVALID-COST POSITIVE ISSUANCE
    # --------------------------------------------------------

    unclassified = (
        ~result[
            "ACV_Computable"
        ]
    )

    result.loc[
        unclassified,
        "ABC_Class",
    ] = "UNCLASSIFIED_COST"

    result.loc[
        unclassified,
        "ABC_Eligibility",
    ] = "UNCLASSIFIED_COST"

    result.loc[
        unclassified,
        [
            "Known_ACV_Total_Year",
            "ACV_Share",
            "Cumulative_ACV",
            "Cumulative_ACV_Share",
            "Cumulative_Share_Before",
        ],
    ] = np.nan

    return result


# ============================================================
# BUILD ALL YEARS
# ============================================================

def build_abc(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

    for year, year_df in (
        fact.groupby(
            "Year",
            sort=True,
        )
    ):

        classified = classify_year(
            year_df
        )

        frames.append(
            classified
        )

    abc = pd.concat(
        frames,
        ignore_index=True,
    )

    return abc


# ============================================================
# VALIDATE ABC
# ============================================================

def validate_abc(
    abc: pd.DataFrame,
) -> None:

    if len(
        abc
    ) != 13665:

        raise ValueError(
            f"Expected 13,665 ABC rows, "
            f"found {len(abc):,}"
        )

    duplicate_count = int(
        abc.duplicated(
            subset=[
                "Medicine_Key",
                "Year",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate Medicine_Key-Year "
            "found in ABC output."
        )

    allowed = {
        "A",
        "B",
        "C",
        "UNCLASSIFIED_COST",
    }

    actual = set(
        abc[
            "ABC_Class"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if not actual.issubset(
        allowed
    ):
        raise ValueError(
            f"Unexpected ABC classes: "
            f"{actual - allowed}"
        )

    eligible_missing = int(
        (
            abc[
                "ACV_Computable"
            ]
            &
            abc[
                "ABC_Class"
            ]
            .isna()
        )
        .sum()
    )

    if eligible_missing:
        raise ValueError(
            f"{eligible_missing} eligible "
            "records lack ABC classification."
        )

    invalid_wrong_class = int(
        (
            ~abc[
                "ACV_Computable"
            ]
            &
            ~abc[
                "ABC_Class"
            ]
            .eq(
                "UNCLASSIFIED_COST"
            )
        )
        .sum()
    )

    if invalid_wrong_class:
        raise ValueError(
            "Invalid-cost records were "
            "incorrectly assigned A/B/C."
        )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    abc: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        abc.groupby(
            "Year",
            sort=True,
        )
    ):

        known_total = float(
            year_df.loc[
                year_df[
                    "ACV_Computable"
                ],
                "Annual_Consumption_Value",
            ]
            .sum()
        )

        for abc_class in [
            "A",
            "B",
            "C",
            "UNCLASSIFIED_COST",
        ]:

            class_df = year_df.loc[
                year_df[
                    "ABC_Class"
                ]
                .eq(
                    abc_class
                )
            ]

            known_class_acv = (
                float(
                    class_df[
                        "Annual_Consumption_Value"
                    ]
                    .dropna()
                    .sum()
                )
            )

            if (
                known_total > 0
                and
                abc_class
                != "UNCLASSIFIED_COST"
            ):
                known_acv_share = (
                    known_class_acv
                    /
                    known_total
                    *
                    100
                )
            else:
                known_acv_share = np.nan

            rows.append(
                {
                    "Year":
                        int(year),

                    "ABC_Class":
                        abc_class,

                    "Medicine_Count":
                        len(class_df),

                    "Active_Issuance_Count":
                        int(
                            class_df[
                                "Annual_Issuance"
                            ]
                            .gt(0)
                            .sum()
                        ),

                    "Known_ACV":
                        known_class_acv,

                    "Known_ACV_Share_Pct":
                        (
                            round(
                                known_acv_share,
                                4,
                            )
                            if not np.isnan(
                                known_acv_share
                            )
                            else np.nan
                        ),

                    "Known_ACV_Total_Year":
                        known_total,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# YEAR-LEVEL COVERAGE SUMMARY
# ============================================================

def build_year_summary(
    abc: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        abc.groupby(
            "Year",
            sort=True,
        )
    ):

        eligible = year_df.loc[
            year_df[
                "ACV_Computable"
            ]
        ]

        unclassified = year_df.loc[
            year_df[
                "ABC_Class"
            ]
            .eq(
                "UNCLASSIFIED_COST"
            )
        ]

        known_acv = float(
            eligible[
                "Annual_Consumption_Value"
            ]
            .sum()
        )

        a_acv = float(
            year_df.loc[
                year_df[
                    "ABC_Class"
                ].eq("A"),
                "Annual_Consumption_Value",
            ]
            .sum()
        )

        ab_acv = float(
            year_df.loc[
                year_df[
                    "ABC_Class"
                ].isin(
                    [
                        "A",
                        "B",
                    ]
                ),
                "Annual_Consumption_Value",
            ]
            .sum()
        )

        rows.append(
            {
                "Year":
                    int(year),

                "Medicine_Rows":
                    len(year_df),

                "ABC_Eligible_Rows":
                    len(eligible),

                "Unclassified_Cost_Rows":
                    len(unclassified),

                "Unclassified_Positive_Issuance":
                    int(
                        unclassified[
                            "Annual_Issuance"
                        ]
                        .gt(0)
                        .sum()
                    ),

                "Known_ACV_Total":
                    known_acv,

                "Category_A_Count":
                    int(
                        year_df[
                            "ABC_Class"
                        ]
                        .eq("A")
                        .sum()
                    ),

                "Category_B_Count":
                    int(
                        year_df[
                            "ABC_Class"
                        ]
                        .eq("B")
                        .sum()
                    ),

                "Category_C_Count":
                    int(
                        year_df[
                            "ABC_Class"
                        ]
                        .eq("C")
                        .sum()
                    ),

                "A_Known_ACV_Coverage_Pct":
                    (
                        round(
                            a_acv
                            / known_acv
                            * 100,
                            4,
                        )
                        if known_acv > 0
                        else np.nan
                    ),

                "AB_Known_ACV_Coverage_Pct":
                    (
                        round(
                            ab_acv
                            / known_acv
                            * 100,
                            4,
                        )
                        if known_acv > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WRITE TO DUCKDB
# ============================================================

def write_database(
    abc: pd.DataFrame,
    summary: pd.DataFrame,
    year_summary: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "abc_df",
            abc,
        )

        connection.register(
            "abc_summary_df",
            summary,
        )

        connection.register(
            "abc_year_summary_df",
            year_summary,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            abc_classification_annual AS
            SELECT *
            FROM abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            abc_summary_annual AS
            SELECT *
            FROM abc_summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            abc_year_summary AS
            SELECT *
            FROM abc_year_summary_df
            """
        )

    finally:
        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 88)
    print("CDH INVENTORY DSS")
    print("ANNUAL ABC ECONOMIC CLASSIFICATION")
    print("=" * 88)
    print()

    fact = load_fact()

    print(
        "Validated medicine-year rows loaded:",
        f"{len(fact):,}",
    )

    abc = build_abc(
        fact
    )

    validate_abc(
        abc
    )

    print(
        "ABC structural validation: PASS"
    )

    summary = build_summary(
        abc
    )

    year_summary = (
        build_year_summary(
            abc
        )
    )

    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    abc_output = (
        OUTPUT_DIR
        / "abc_classification_annual.parquet"
    )

    summary_output = (
        OUTPUT_DIR
        / "abc_summary_annual.csv"
    )

    year_output = (
        OUTPUT_DIR
        / "abc_year_summary.csv"
    )

    abc.to_parquet(
        abc_output,
        index=False,
        engine="pyarrow",
    )

    summary.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    year_summary.to_csv(
        year_output,
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        abc,
        summary,
        year_summary,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 88)
    print("YEAR-LEVEL ABC RESULTS")
    print("-" * 88)

    print(
        year_summary
        .to_string(
            index=False
        )
    )

    print()
    print("-" * 88)
    print("ABC CLASS SUMMARY")
    print("-" * 88)

    print(
        summary
        .to_string(
            index=False
        )
    )

    print()
    print("-" * 88)
    print("METHOD RULES")
    print("-" * 88)

    print(
        "A threshold: 70% cumulative "
        "KNOWN annual consumption value."
    )

    print(
        "B threshold: through 90% cumulative "
        "KNOWN annual consumption value."
    )

    print(
        "Boundary-crossing medicine remains "
        "in the higher-priority category."
    )

    print(
        "Positive-issuance records with "
        "invalid historical cost are "
        "UNCLASSIFIED_COST."
    )

    print(
        "ABC represents economic priority, "
        "not clinical criticality."
    )

    print()
    print("=" * 88)
    print(
        "ABC CLASSIFICATION COMPLETE"
    )
    print("=" * 88)

    print(
        "No validated historical source "
        "record was modified."
    )


if __name__ == "__main__":
    main()
    