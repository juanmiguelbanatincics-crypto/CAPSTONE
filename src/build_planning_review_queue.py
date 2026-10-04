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
    / "planning"
    / "planning_2026_review_v1"
)

OVERRIDE_DIR = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "planning_overrides"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OVERRIDE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

REVIEW_VERSION = "PLANNING_2026_REVIEW_V1"

EXPECTED_MEDICINES = 2733

ABC_PRIORITY = {
    "A": 1,
    "B": 2,
    "UNCLASSIFIED_COST": 3,
    "C": 4,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_planning_inputs() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            """
            SELECT *
            FROM planning_2026_inputs
            ORDER BY Medicine_Key
            """
        ).df()

    finally:

        connection.close()

    return dataframe


# ============================================================
# VALIDATE SOURCE
# ============================================================

def validate_source(
    dataframe: pd.DataFrame,
) -> None:

    if len(dataframe) != EXPECTED_MEDICINES:

        raise ValueError(
            f"Expected {EXPECTED_MEDICINES:,} "
            f"planning rows; found "
            f"{len(dataframe):,}"
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .nunique()
        != EXPECTED_MEDICINES
    ):

        raise ValueError(
            "Planning Medicine_Key "
            "uniqueness failed."
        )


# ============================================================
# BUILD REVIEW QUEUE
# ============================================================

def build_review_queue(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    review = (
        dataframe.loc[
            ~dataframe[
                "Optimizer_Core_Input_Ready"
            ]
            .fillna(False)
            .astype(bool)
        ]
        .copy()
    )

    review[
        "Needs_Stock_Review"
    ] = (
        ~review[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    review[
        "Needs_Cost_Review"
    ] = (
        ~review[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    review[
        "ABC_Priority_Order"
    ] = (
        review[
            "ABC_AsOf"
        ]
        .map(
            ABC_PRIORITY
        )
        .fillna(99)
        .astype(int)
    )

    # --------------------------------------------------------
    # REQUIREMENT EXPOSURE
    # --------------------------------------------------------

    review[
        "Maximum_Planning_Requirement"
    ] = (
        review[
            [
                "Requirement_Point",
                "Requirement_P80",
                "Requirement_P90",
            ]
        ]
        .max(
            axis=1
        )
    )

    review[
        "Has_Positive_Point_Requirement"
    ] = (
        review[
            "Requirement_Point"
        ]
        .gt(0)
    )

    review[
        "Has_Positive_P80_Requirement"
    ] = (
        review[
            "Requirement_P80"
        ]
        .gt(0)
    )

    review[
        "Has_Positive_P90_Requirement"
    ] = (
        review[
            "Requirement_P90"
        ]
        .gt(0)
    )

    review[
        "High_Priority_ABC_Review"
    ] = (
        review[
            "ABC_AsOf"
        ]
        .isin(
            [
                "A",
                "B",
            ]
        )
    )

    review[
        "Full_Optimization_Blocker"
    ] = (
        review[
            "High_Priority_ABC_Review"
        ]
    )

    review[
        "Review_Version"
    ] = REVIEW_VERSION

    keep_columns = [
        "Medicine_Key",
        "Brand_Name",
        "Generic_Name",
        "Unit",
        "ABC_AsOf",
        "Behavior_Segment_AsOf",
        "Planning_Input_Status",
        "Planning_Stock_Status",
        "Planning_Cost_Status",
        "Needs_Stock_Review",
        "Needs_Cost_Review",
        "High_Priority_ABC_Review",
        "Full_Optimization_Blocker",
        "Source_Ending_Balance_2025",
        "Planning_Stock_Qty",
        "Planning_Cost",
        "Planning_Cost_Source_Year",
        "Planning_Cost_Age_Years",
        "Requirement_Point",
        "Requirement_P80",
        "Requirement_P90",
        "Maximum_Planning_Requirement",
        "Has_Positive_Point_Requirement",
        "Has_Positive_P80_Requirement",
        "Has_Positive_P90_Requirement",
        "Flag_P80_Below_Point",
        "Flag_P90_Below_Point",
        "ABC_Priority_Order",
        "Review_Version",
    ]

    review = (
        review[
            keep_columns
        ]
        .sort_values(
            [
                "ABC_Priority_Order",
                "Full_Optimization_Blocker",
                "Maximum_Planning_Requirement",
                "Medicine_Key",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    review[
        "Review_Queue_Order"
    ] = (
        np.arange(
            1,
            len(review) + 1,
        )
    )

    return review


# ============================================================
# BUILD BLANK OVERRIDE TEMPLATE
# ============================================================

def build_override_template(
    review: pd.DataFrame,
) -> pd.DataFrame:

    template = review[
        [
            "Review_Queue_Order",
            "Medicine_Key",
            "Brand_Name",
            "Generic_Name",
            "Unit",
            "ABC_AsOf",
            "Behavior_Segment_AsOf",
            "Planning_Input_Status",
            "Planning_Stock_Status",
            "Planning_Cost_Status",
            "Needs_Stock_Review",
            "Needs_Cost_Review",
            "Requirement_Point",
            "Requirement_P80",
            "Requirement_P90",
        ]
    ].copy()

    # --------------------------------------------------------
    # USER-SUPPLIED FIELDS
    #
    # These are intentionally blank.
    # The system does not invent values.
    # --------------------------------------------------------

    template[
        "Override_Stock_Qty"
    ] = np.nan

    template[
        "Override_Stock_AsOf_Date"
    ] = ""

    template[
        "Override_Stock_Source"
    ] = ""

    template[
        "Override_Planning_Cost"
    ] = np.nan

    template[
        "Override_Cost_AsOf_Date"
    ] = ""

    template[
        "Override_Cost_Source"
    ] = ""

    template[
        "Reviewer_Name_or_ID"
    ] = ""

    template[
        "Review_Notes"
    ] = ""

    template[
        "Approval_Status"
    ] = "PENDING"

    template[
        "Override_Template_Version"
    ] = REVIEW_VERSION

    return template


# ============================================================
# REVIEW SUMMARY
# ============================================================

def build_review_summary(
    review: pd.DataFrame,
) -> pd.DataFrame:

    row = {
        "Review_Row_Count":
            len(review),

        "Stock_Review_Count":
            int(
                review[
                    "Needs_Stock_Review"
                ]
                .sum()
            ),

        "Cost_Review_Count":
            int(
                review[
                    "Needs_Cost_Review"
                ]
                .sum()
            ),

        "High_Priority_AB_Blocker_Count":
            int(
                review[
                    "Full_Optimization_Blocker"
                ]
                .sum()
            ),

        "Positive_Point_Requirement_Review_Count":
            int(
                review[
                    "Has_Positive_Point_Requirement"
                ]
                .sum()
            ),

        "Positive_P80_Requirement_Review_Count":
            int(
                review[
                    "Has_Positive_P80_Requirement"
                ]
                .sum()
            ),

        "Positive_P90_Requirement_Review_Count":
            int(
                review[
                    "Has_Positive_P90_Requirement"
                ]
                .sum()
            ),

        "Review_Point_Requirement_Total":
            float(
                review[
                    "Requirement_Point"
                ]
                .sum()
            ),

        "Review_P80_Requirement_Total":
            float(
                review[
                    "Requirement_P80"
                ]
                .sum()
            ),

        "Review_P90_Requirement_Total":
            float(
                review[
                    "Requirement_P90"
                ]
                .sum()
            ),

        "Review_Version":
            REVIEW_VERSION,
    }

    return pd.DataFrame(
        [row]
    )


# ============================================================
# REVIEW BY ABC
# ============================================================

def build_review_by_abc(
    review: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for abc_class, group in (
        review.groupby(
            "ABC_AsOf",
            dropna=False,
            sort=True,
        )
    ):

        abc_name = (
            str(abc_class)
            if pd.notna(
                abc_class
            )
            else "MISSING"
        )

        rows.append(
            {
                "ABC_AsOf":
                    abc_name,

                "Review_Row_Count":
                    len(group),

                "Stock_Review_Count":
                    int(
                        group[
                            "Needs_Stock_Review"
                        ]
                        .sum()
                    ),

                "Cost_Review_Count":
                    int(
                        group[
                            "Needs_Cost_Review"
                        ]
                        .sum()
                    ),

                "Point_Requirement_Total":
                    float(
                        group[
                            "Requirement_Point"
                        ]
                        .sum()
                    ),

                "P80_Requirement_Total":
                    float(
                        group[
                            "Requirement_P80"
                        ]
                        .sum()
                    ),

                "P90_Requirement_Total":
                    float(
                        group[
                            "Requirement_P90"
                        ]
                        .sum()
                    ),

                "P80_Below_Point_Count":
                    int(
                        group[
                            "Flag_P80_Below_Point"
                        ]
                        .sum()
                    ),

                "P90_Below_Point_Count":
                    int(
                        group[
                            "Flag_P90_Below_Point"
                        ]
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# QUANTILE / POINT DISAGREEMENT SUMMARY
# ============================================================

def build_requirement_disagreement(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for abc_class, group in (
        dataframe.groupby(
            "ABC_AsOf",
            dropna=False,
            sort=True,
        )
    ):

        abc_name = (
            str(abc_class)
            if pd.notna(
                abc_class
            )
            else "MISSING"
        )

        rows.append(
            {
                "ABC_AsOf":
                    abc_name,

                "Medicine_Count":
                    len(group),

                "P80_Below_Point_Count":
                    int(
                        group[
                            "Flag_P80_Below_Point"
                        ]
                        .sum()
                    ),

                "P80_Below_Point_Pct":
                    float(
                        group[
                            "Flag_P80_Below_Point"
                        ]
                        .mean()
                        * 100
                    ),

                "P90_Below_Point_Count":
                    int(
                        group[
                            "Flag_P90_Below_Point"
                        ]
                        .sum()
                    ),

                "P90_Below_Point_Pct":
                    float(
                        group[
                            "Flag_P90_Below_Point"
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
# VALIDATION
# ============================================================

def validate_review(
    source: pd.DataFrame,
    review: pd.DataFrame,
    template: pd.DataFrame,
) -> None:

    expected_review = int(
        (
            ~source[
                "Optimizer_Core_Input_Ready"
            ]
            .fillna(False)
            .astype(bool)
        )
        .sum()
    )

    if len(
        review
    ) != expected_review:

        raise ValueError(
            f"Expected {expected_review:,} "
            f"review rows; found "
            f"{len(review):,}"
        )

    if len(
        template
    ) != expected_review:

        raise ValueError(
            "Override-template row count "
            "does not match review queue."
        )

    if (
        review[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate Medicine_Key "
            "in review queue."
        )

    if not (
        template[
            "Approval_Status"
        ]
        .eq(
            "PENDING"
        )
        .all()
    ):

        raise ValueError(
            "Generated override template "
            "must start as PENDING."
        )

    if (
        template[
            "Override_Stock_Qty"
        ]
        .notna()
        .any()
    ):

        raise ValueError(
            "Generated stock overrides "
            "must be blank."
        )

    if (
        template[
            "Override_Planning_Cost"
        ]
        .notna()
        .any()
    ):

        raise ValueError(
            "Generated cost overrides "
            "must be blank."
        )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    review: pd.DataFrame,
    summary: pd.DataFrame,
    by_abc: pd.DataFrame,
    disagreement: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "review_df",
            review,
        )

        connection.register(
            "summary_df",
            summary,
        )

        connection.register(
            "by_abc_df",
            by_abc,
        )

        connection.register(
            "disagreement_df",
            disagreement,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_review_queue AS
            SELECT *
            FROM review_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_review_summary AS
            SELECT *
            FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_review_by_abc AS
            SELECT *
            FROM by_abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_requirement_disagreement AS
            SELECT *
            FROM disagreement_df
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
    print("2026 CONTROLLED PLANNING REVIEW QUEUE")
    print("=" * 108)
    print()

    planning = (
        load_planning_inputs()
    )

    validate_source(
        planning
    )

    print(
        "Planning source validation: PASS"
    )

    review = (
        build_review_queue(
            planning
        )
    )

    template = (
        build_override_template(
            review
        )
    )

    validate_review(
        planning,
        review,
        template,
    )

    print(
        "Review queue structural "
        "validation: PASS"
    )

    summary = (
        build_review_summary(
            review
        )
    )

    by_abc = (
        build_review_by_abc(
            review
        )
    )

    disagreement = (
        build_requirement_disagreement(
            planning
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    review.to_csv(
        OUTPUT_DIR
        / "planning_2026_review_queue.csv",
        index=False,
        encoding="utf-8-sig",
    )

    review.to_parquet(
        OUTPUT_DIR
        / "planning_2026_review_queue.parquet",
        index=False,
        engine="pyarrow",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "planning_2026_review_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "planning_2026_review_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    disagreement.to_csv(
        OUTPUT_DIR
        / "planning_2026_requirement_disagreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    template_path = (
        OVERRIDE_DIR
        / "planning_2026_override_template.csv"
    )

    template.to_csv(
        template_path,
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        review=review,
        summary=summary,
        by_abc=by_abc,
        disagreement=disagreement,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 108)
    print("REVIEW SUMMARY")
    print("-" * 108)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("REVIEW BY 2025 ABC")
    print("-" * 108)

    print(
        by_abc.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("P80 / P90 BELOW INCUMBENT POINT FORECAST")
    print("-" * 108)

    print(
        disagreement.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("HIGH-PRIORITY A/B REVIEW ITEMS")
    print("-" * 108)

    high_priority = (
        review.loc[
            review[
                "High_Priority_ABC_Review"
            ]
        ]
    )

    if high_priority.empty:

        print(
            "None."
        )

    else:

        print(
            high_priority[
                [
                    "Review_Queue_Order",
                    "Medicine_Key",
                    "Brand_Name",
                    "ABC_AsOf",
                    "Behavior_Segment_AsOf",
                    "Planning_Stock_Status",
                    "Planning_Cost_Status",
                    "Requirement_Point",
                    "Requirement_P80",
                    "Requirement_P90",
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
    print(
        "Blank override template created:"
    )

    print(
        template_path
    )

    print()
    print("-" * 108)
    print("CONTROL RULE")
    print("-" * 108)

    print(
        "No stock or cost override value "
        "was generated automatically."
    )

    print(
        "Every override row starts PENDING."
    )

    print(
        "Missing/negative source stock "
        "remains unresolved until an "
        "authorized value is supplied."
    )

    print(
        "Missing planning cost remains "
        "unresolved until an authorized "
        "planning cost is supplied."
    )

    print(
        "The presence of an unresolved A/B "
        "medicine blocks a claim that a "
        "FULL A-to-B-to-C optimization "
        "covers the complete high-priority "
        "medicine set."
    )

    print()
    print("=" * 108)
    print(
        "PLANNING REVIEW QUEUE COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()