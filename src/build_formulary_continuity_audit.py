from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

try:
    from .runtime_context import load_runtime_context_from_env
except ImportError:
    from runtime_context import load_runtime_context_from_env


RUNTIME_CONTEXT = load_runtime_context_from_env()

DATABASE_PATH = RUNTIME_CONTEXT.database_path
OUTPUT_DIR = RUNTIME_CONTEXT.validated_annual_dir

LATEST_YEAR = (
    RUNTIME_CONTEXT.latest_completed_year
)

PREVIOUS_YEAR = (
    LATEST_YEAR - 1
)


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
                Unit
            FROM fact_inventory_annual
            ORDER BY Medicine_Key, Year
            """
        ).df()

    finally:
        connection.close()

    return fact


def build_continuity_audit(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    observed_years = set(
        fact["Year"]
        .astype(int)
        .unique()
        .tolist()
    )

    required_years = {
        PREVIOUS_YEAR,
        LATEST_YEAR,
    }

    missing_years = sorted(
        required_years
        - observed_years
    )

    if missing_years:
        raise ValueError(
            "Continuity audit requires both "
            f"{PREVIOUS_YEAR} and {LATEST_YEAR}. "
            f"Missing: {missing_years}"
        )

    duplicate_count = int(
        fact.duplicated(
            subset=[
                "Medicine_Key",
                "Year",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Duplicate Medicine_Key-Year "
            f"records found: {duplicate_count}"
        )

    latest_identity = (
        fact.sort_values(
            [
                "Medicine_Key",
                "Year",
            ]
        )
        .groupby(
            "Medicine_Key",
            as_index=False,
        )
        .tail(1)
        [
            [
                "Medicine_Key",
                "Brand_Name",
                "Generic_Name",
                "Unit",
            ]
        ]
    )

    history = (
        fact.groupby(
            "Medicine_Key",
            as_index=False,
        )
        .agg(
            First_Observed_Year=(
                "Year",
                "min",
            ),
            Last_Observed_Year=(
                "Year",
                "max",
            ),
            Years_Observed=(
                "Year",
                "nunique",
            ),
        )
    )

    previous_keys = set(
        fact.loc[
            fact["Year"].eq(
                PREVIOUS_YEAR
            ),
            "Medicine_Key",
        ]
    )

    latest_keys = set(
        fact.loc[
            fact["Year"].eq(
                LATEST_YEAR
            ),
            "Medicine_Key",
        ]
    )

    audit = history.merge(
        latest_identity,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    audit[
        "Previous_Year"
    ] = PREVIOUS_YEAR

    audit[
        "Latest_Completed_Year"
    ] = LATEST_YEAR

    audit[
        "Present_Previous_Year"
    ] = audit[
        "Medicine_Key"
    ].isin(
        previous_keys
    )

    audit[
        "Present_Latest_Year"
    ] = audit[
        "Medicine_Key"
    ].isin(
        latest_keys
    )

    audit[
        "Continuity_Status"
    ] = "HISTORICAL_NOT_RECENT"

    continuing = (
        audit["Present_Previous_Year"]
        &
        audit["Present_Latest_Year"]
    )

    new_latest = (
        ~audit["Present_Previous_Year"]
        &
        audit["Present_Latest_Year"]
        &
        audit["First_Observed_Year"].eq(
            LATEST_YEAR
        )
    )

    returning = (
        ~audit["Present_Previous_Year"]
        &
        audit["Present_Latest_Year"]
        &
        audit["First_Observed_Year"].lt(
            LATEST_YEAR
        )
    )

    absent_latest = (
        audit["Present_Previous_Year"]
        &
        ~audit["Present_Latest_Year"]
    )

    audit.loc[
        continuing,
        "Continuity_Status",
    ] = "CONTINUING"

    audit.loc[
        new_latest,
        "Continuity_Status",
    ] = "NEW_IN_LATEST_YEAR"

    audit.loc[
        returning,
        "Continuity_Status",
    ] = "RETURNING_AFTER_GAP"

    audit.loc[
        absent_latest,
        "Continuity_Status",
    ] = "ABSENT_IN_LATEST_YEAR"

    audit[
        "Planning_Population_Latest"
    ] = audit[
        "Present_Latest_Year"
    ]

    audit[
        "Continuity_Review_Required"
    ] = audit[
        "Continuity_Status"
    ].isin(
        [
            "NEW_IN_LATEST_YEAR",
            "RETURNING_AFTER_GAP",
            "ABSENT_IN_LATEST_YEAR",
        ]
    )

    audit[
        "Review_Reason"
    ] = "NONE"

    audit.loc[
        audit[
            "Continuity_Status"
        ].eq(
            "NEW_IN_LATEST_YEAR"
        ),
        "Review_Reason",
    ] = "LIMITED_HISTORY"

    audit.loc[
        audit[
            "Continuity_Status"
        ].eq(
            "RETURNING_AFTER_GAP"
        ),
        "Review_Reason",
    ] = (
        "IDENTITY_OR_FORMULARY_REVIEW"
    )

    audit.loc[
        audit[
            "Continuity_Status"
        ].eq(
            "ABSENT_IN_LATEST_YEAR"
        ),
        "Review_Reason",
    ] = (
        "FORMULARY_CONTINUITY_REVIEW"
    )

    return (
        audit.sort_values(
            [
                "Continuity_Status",
                "Brand_Name",
                "Medicine_Key",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )



# ============================================================
# WRITE GOVERNED AUDIT TABLES
# ============================================================

def write_database(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:

    if not RUNTIME_CONTEXT.is_isolated_run:
        raise RuntimeError(
            "Continuity audit database writes are "
            "allowed only in TEST or STAGING mode."
        )

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "continuity_audit_df",
            audit,
        )

        connection.register(
            "continuity_summary_df",
            summary,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            formulary_continuity_audit AS
            SELECT *
            FROM continuity_audit_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            formulary_continuity_summary AS
            SELECT *
            FROM continuity_summary_df
            """
        )

    finally:
        connection.close()

def main() -> None:

    if not (
        RUNTIME_CONTEXT.is_isolated_run
    ):
        raise RuntimeError(
            "Development continuity audit may "
            "currently run only in TEST or "
            "STAGING mode."
        )

    print("=" * 88)
    print("CDH INVENTORY DSS")
    print("FORMULARY CONTINUITY AUDIT")
    print("=" * 88)
    print()

    fact = load_fact()

    audit = build_continuity_audit(
        fact
    )

    summary = (
        audit.groupby(
            "Continuity_Status",
            dropna=False,
        )
        .size()
        .rename("Medicine_Count")
        .reset_index()
        .sort_values(
            "Continuity_Status"
        )
    )

    summary[
        "Previous_Year"
    ] = PREVIOUS_YEAR

    summary[
        "Latest_Completed_Year"
    ] = LATEST_YEAR

    summary[
        "Run_Mode"
    ] = RUNTIME_CONTEXT.run_mode

    summary[
        "Run_ID"
    ] = RUNTIME_CONTEXT.run_id

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "formulary_continuity_audit.csv"
    )

    audit.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_path = (
        OUTPUT_DIR
        / "formulary_continuity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        audit,
        summary,
    )

    print(
        f"Previous year: {PREVIOUS_YEAR}"
    )

    print(
        f"Latest completed year: {LATEST_YEAR}"
    )

    print(
        f"Unique medicines across history: "
        f"{audit['Medicine_Key'].nunique():,}"
    )

    print()
    print(summary.to_string(index=False))

    print()
    print(
        "Review-required medicines:",
        int(
            audit[
                "Continuity_Review_Required"
            ].sum()
        ),
    )

    print()
    print(
        "Audit output:",
        output_path,
    )

    print(
        "Summary output:",
        summary_path,
    )

    print()
    print(
        "No missing medicine-year records "
        "were fabricated."
    )

    print(
        "Governed continuity tables were "
        "written only to the isolated "
        f"{RUNTIME_CONTEXT.run_mode} database."
    )


if __name__ == "__main__":
    main()
