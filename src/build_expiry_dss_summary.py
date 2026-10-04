from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# PATHS / VERSION
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "cdh_inventory.duckdb"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expiry"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_VERSION = (
    "EXPIRY_DSS_MEDICINE_SUMMARY_V1"
)


# ============================================================
# LOAD
# ============================================================

def load_inputs():

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        planning = connection.execute(
            """
            SELECT *
            FROM planning_2026_inputs
            """
        ).df()

        expiry = connection.execute(
            """
            SELECT *
            FROM fact_expiry_medicine_records
            WHERE Mapping_Status =
                  'AUTHORITATIVE_EXACT_MATCH'
            """
        ).df()

        review_queue = connection.execute(
            """
            SELECT *
            FROM expiry_medicine_review_queue
            """
        ).df()

        validation = connection.execute(
            """
            SELECT *
            FROM expiry_validation_summary
            """
        ).df()

        source_governance = (
            connection.execute(
                """
                SELECT *
                FROM expiry_source_governance
                """
            ).df()
        )

    finally:

        connection.close()

    return (
        planning,
        expiry,
        review_queue,
        validation,
        source_governance,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(
    planning: pd.DataFrame,
    expiry: pd.DataFrame,
    review_queue: pd.DataFrame,
    validation: pd.DataFrame,
    source_governance: pd.DataFrame,
) -> None:

    if planning.empty:

        raise ValueError(
            "planning_2026_inputs is empty."
        )

    if expiry.empty:

        raise ValueError(
            "Authoritative expiry fact is empty."
        )

    if len(validation) != 1:

        raise ValueError(
            "Expected one expiry validation row."
        )

    if len(source_governance) != 1:

        raise ValueError(
            "Expected one expiry governance row."
        )

    if (
        planning["Medicine_Key"]
        .isna()
        .any()
    ):

        raise ValueError(
            "Planning input contains null "
            "Medicine_Key."
        )

    if (
        planning["Medicine_Key"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "planning_2026_inputs is not "
            "one row per Medicine_Key."
        )

    if not (
        planning["Target_Year"]
        .eq(2026)
        .all()
    ):

        raise ValueError(
            "Unexpected planning target year."
        )

    if (
        expiry["Medicine_Key"]
        .isna()
        .any()
    ):

        raise ValueError(
            "Authoritative expiry records "
            "contain null Medicine_Key."
        )

    if not (
        expiry["Mapping_Status"]
        .eq(
            "AUTHORITATIVE_EXACT_MATCH"
        )
        .all()
    ):

        raise ValueError(
            "Non-authoritative expiry mapping "
            "entered DSS summary input."
        )

    item_per_medicine = (
        expiry.groupby(
            "Medicine_Key"
        )["Item_ID"]
        .nunique()
    )

    if (
        item_per_medicine > 1
    ).any():

        raise ValueError(
            "A Medicine_Key maps to more than "
            "one expiry Item_ID."
        )

    medicine_per_item = (
        expiry.groupby(
            "Item_ID"
        )["Medicine_Key"]
        .nunique()
    )

    if (
        medicine_per_item > 1
    ).any():

        raise ValueError(
            "An expiry Item_ID maps to more "
            "than one Medicine_Key."
        )

    if bool(
        source_governance.loc[
            0,
            "Current_Lot_Level_"
            "Expiry_Exposure_Computable",
        ]
    ):

        raise ValueError(
            "Current lot-level expiry exposure "
            "must remain unsupported."
        )

    if bool(
        source_governance.loc[
            0,
            "Source_Quantity_Interpreted_"
            "As_Current_On_Hand",
        ]
    ):

        raise ValueError(
            "Expiry quantity must not be "
            "treated as current on-hand."
        )

    if bool(
        source_governance.loc[
            0,
            "Source_Net_Amount_Interpreted_"
            "As_Current_Expiry_Exposure",
        ]
    ):

        raise ValueError(
            "Source Net_Amount must not be "
            "treated as current expiry exposure."
        )

    if bool(
        source_governance.loc[
            0,
            "Actual_Expired_Waste_Claim",
        ]
    ):

        raise ValueError(
            "Actual expired waste claim "
            "must remain false."
        )

    expected_review = int(
        validation.loc[
            0,
            "Unresolved_Medicine_Item_ID",
        ]
    )

    if len(review_queue) != expected_review:

        raise ValueError(
            "Expiry review queue does not "
            "reconcile to validation summary."
        )


# ============================================================
# BUILD AGGREGATED HISTORY
# ============================================================

def build_history_summary(
    expiry: pd.DataFrame,
) -> pd.DataFrame:

    data = expiry.copy()

    data["Lot_Null_Or_Blank"] = (
        data["lotno"].isna()
        |
        data["lotno"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    data["Lot_Zero_Placeholder"] = (
        data["lotno"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("0")
    )

    data["Usable_Lot_Record"] = (
        ~data["Lot_Null_Or_Blank"]
        &
        ~data["Lot_Zero_Placeholder"]
    )

    data["Usable_Lot_Value"] = (
        data["lotno"]
        .where(
            data["Usable_Lot_Record"]
        )
    )

    summary = (
        data.groupby(
            "Medicine_Key",
            dropna=False,
        )
        .agg(
            Expiry_Item_ID=(
                "Item_ID",
                "first",
            ),

            Expiry_History_Record_Count=(
                "Source_Row_Number",
                "size",
            ),

            Expiry_History_Document_Count=(
                "Document_No_",
                "nunique",
            ),

            Earliest_Expiry_Document_Date=(
                "Document_Date",
                "min",
            ),

            Latest_Expiry_Document_Date=(
                "Document_Date",
                "max",
            ),

            Earliest_Expiration_Date_Represented=(
                "Expiration_Date",
                "min",
            ),

            Latest_Expiration_Date_Represented=(
                "Expiration_Date",
                "max",
            ),

            Earliest_Expiration_Year_Represented=(
                "Expiration_Year",
                "min",
            ),

            Latest_Expiration_Year_Represented=(
                "Expiration_Year",
                "max",
            ),

            Expiration_Year_Count=(
                "Expiration_Year",
                "nunique",
            ),

            Usable_Lot_Record_Count=(
                "Usable_Lot_Record",
                "sum",
            ),

            Null_Or_Blank_Lot_Record_Count=(
                "Lot_Null_Or_Blank",
                "sum",
            ),

            Zero_Placeholder_Lot_Record_Count=(
                "Lot_Zero_Placeholder",
                "sum",
            ),

            Distinct_Usable_Lot_Value_Count=(
                "Usable_Lot_Value",
                "nunique",
            ),

            Duplicate_Flagged_Record_Count=(
                "Flag_Exact_Source_Duplicate",
                "sum",
            ),

            Document_After_Expiration_Record_Count=(
                "Flag_Document_After_Expiration",
                "sum",
            ),

            Net_Arithmetic_Mismatch_Record_Count=(
                "Flag_Net_Amount_Arithmetic_Mismatch",
                "sum",
            ),

            Zero_Land_Cost_Record_Count=(
                "Flag_Zero_Land_Cost",
                "sum",
            ),

            Source_Review_Record_Count=(
                "Data_Quality_Status",
                lambda x: int(
                    x.eq("REVIEW").sum()
                ),
            ),
        )
        .reset_index()
    )

    return summary


# ============================================================
# BUILD DSS MEDICINE SUMMARY
# ============================================================

def build_dss_summary(
    planning: pd.DataFrame,
    expiry: pd.DataFrame,
    validation: pd.DataFrame,
    source_governance: pd.DataFrame,
) -> pd.DataFrame:

    history = build_history_summary(
        expiry
    )

    planning_columns = [
        "Medicine_Key",
        "Brand_Name",
        "Generic_Name",
        "Unit",
        "Target_Year",
        "ABC_AsOf",
        "Behavior_Segment_AsOf",
        "Point_Forecast_2026",
        "Source_Ending_Balance_2025",
        "Planning_Stock_Status",
        "Planning_Stock_Valid",
        "Planning_Stock_Qty",
        "Planning_Cost",
        "Planning_Cost_Available",
        "Optimizer_Core_Input_Ready",
        "Planning_Input_Status",
    ]

    missing = [
        column
        for column in planning_columns
        if column not in planning.columns
    ]

    if missing:

        raise ValueError(
            "Missing planning columns: "
            f"{missing}"
        )

    summary = (
        planning[
            planning_columns
        ]
        .merge(
            history,
            on="Medicine_Key",
            how="left",
            validate="one_to_one",
        )
    )

    summary[
        "Has_Authoritative_Expiry_History"
    ] = (
        summary[
            "Expiry_History_Record_Count"
        ]
        .notna()
    )

    integer_count_columns = [
        "Expiry_History_Record_Count",
        "Expiry_History_Document_Count",
        "Earliest_Expiration_Year_Represented",
        "Latest_Expiration_Year_Represented",
        "Expiration_Year_Count",
        "Usable_Lot_Record_Count",
        "Null_Or_Blank_Lot_Record_Count",
        "Zero_Placeholder_Lot_Record_Count",
        "Distinct_Usable_Lot_Value_Count",
        "Duplicate_Flagged_Record_Count",
        "Document_After_Expiration_Record_Count",
        "Net_Arithmetic_Mismatch_Record_Count",
        "Zero_Land_Cost_Record_Count",
        "Source_Review_Record_Count",
    ]

    count_only_columns = [
        column
        for column in integer_count_columns
        if "Year_Represented" not in column
    ]

    for column in count_only_columns:

        summary[column] = (
            summary[column]
            .fillna(0)
            .astype("int64")
        )

    for column in [
        "Earliest_Expiration_Year_Represented",
        "Latest_Expiration_Year_Represented",
    ]:

        summary[column] = (
            summary[column]
            .astype("Int64")
        )

    summary[
        "Expiry_History_Coverage_Status"
    ] = np.where(
        summary[
            "Has_Authoritative_Expiry_History"
        ],
        "LINKED_HISTORICAL_EXPIRY_EVIDENCE",
        "NO_LINKED_EXPIRY_HISTORY_IN_EXTRACT",
    )

    has_quality_issue = (
        summary[
            "Source_Review_Record_Count"
        ]
        > 0
    )

    summary[
        "Expiry_History_Data_Quality_Status"
    ] = np.select(
        [
            ~summary[
                "Has_Authoritative_Expiry_History"
            ],
            has_quality_issue,
        ],
        [
            "NOT_APPLICABLE_NO_LINKED_HISTORY",
            "REVIEW",
        ],
        default="PASS",
    )

    observed_min = (
        pd.to_datetime(
            validation.loc[
                0,
                "Observed_Expiration_Min",
            ]
        )
    )

    observed_max = (
        pd.to_datetime(
            validation.loc[
                0,
                "Observed_Expiration_Max",
            ]
        )
    )

    report_to = (
        pd.to_datetime(
            validation.loc[
                0,
                "Report_Expiration_To",
            ]
        )
    )

    live_forward_supported = bool(
        source_governance.loc[
            0,
            "Live_Forward_Expiry_Alert_"
            "Supported_As_Of_Build",
        ]
    )

    summary[
        "Expiry_Source_Observed_Min"
    ] = observed_min

    summary[
        "Expiry_Source_Observed_Max"
    ] = observed_max

    summary[
        "Expiry_Source_Report_To"
    ] = report_to

    summary[
        "Expiry_Source_Live_Forward_Alert_Supported"
    ] = live_forward_supported

    summary[
        "Planning_Stock_Linked_To_Expiry_Lot"
    ] = False

    summary[
        "Current_Lot_Level_Expiry_Exposure_Computable"
    ] = False

    summary[
        "Current_Expiring_Units_Computable"
    ] = False

    summary[
        "Current_Expiry_Value_Computable"
    ] = False

    summary[
        "Actual_Expired_Waste_Claim"
    ] = False

    summary[
        "Linked_History_Does_Not_Imply_Current_Risk"
    ] = True

    summary[
        "No_Linked_History_Does_Not_Imply_No_Risk"
    ] = True

    summary[
        "Expiry_DSS_Interpretation"
    ] = np.where(
        summary[
            "Has_Authoritative_Expiry_History"
        ],
        "HISTORICAL_EXPIRY_EVIDENCE_ONLY",
        "NO_LINKED_EXPIRY_HISTORY_IN_SOURCE_EXTRACT",
    )

    summary[
        "Expiry_Summary_Version"
    ] = SUMMARY_VERSION

    return summary


# ============================================================
# GOVERNANCE SUMMARY
# ============================================================

def build_governance(
    summary: pd.DataFrame,
    expiry: pd.DataFrame,
    review_queue: pd.DataFrame,
    validation: pd.DataFrame,
    source_governance: pd.DataFrame,
) -> pd.DataFrame:

    linked = int(
        summary[
            "Has_Authoritative_Expiry_History"
        ].sum()
    )

    total = len(summary)

    lot_null = int(
        expiry["lotno"]
        .isna()
        .sum()
        +
        expiry["lotno"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    # Avoid double-counting true nulls because
    # fillna("") also satisfies the blank test.
    lot_null = int(
        (
            expiry["lotno"].isna()
            |
            expiry["lotno"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        ).sum()
    )

    lot_zero = int(
        expiry["lotno"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("0")
        .sum()
    )

    usable_lot = (
        len(expiry)
        -
        lot_null
        -
        lot_zero
    )

    build_timestamp = (
        datetime.now(timezone.utc)
        .replace(tzinfo=None)
    )

    governance = pd.DataFrame(
        [
            {
                "Build_Timestamp_UTC":
                    build_timestamp,

                "Expiry_Summary_Version":
                    SUMMARY_VERSION,

                "Planning_Target_Year":
                    2026,

                "Planning_Medicine_Count":
                    total,

                "Medicines_With_Authoritative_Expiry_History":
                    linked,

                "Medicines_Without_Linked_Expiry_History":
                    total - linked,

                "Authoritative_Expiry_History_Coverage_Rate":
                    linked / total,

                "Authoritative_Expiry_Record_Count":
                    int(len(expiry)),

                "Unresolved_Expiry_Item_ID_Review_Count":
                    int(len(review_queue)),

                "Usable_Lot_Record_Count":
                    int(usable_lot),

                "Null_Or_Blank_Lot_Record_Count":
                    lot_null,

                "Zero_Placeholder_Lot_Record_Count":
                    lot_zero,

                "Observed_Expiration_Min":
                    validation.loc[
                        0,
                        "Observed_Expiration_Min",
                    ],

                "Observed_Expiration_Max":
                    validation.loc[
                        0,
                        "Observed_Expiration_Max",
                    ],

                "Live_Forward_Expiry_Alert_Supported":
                    bool(
                        source_governance.loc[
                            0,
                            "Live_Forward_Expiry_Alert_"
                            "Supported_As_Of_Build",
                        ]
                    ),

                "Planning_Stock_Linked_To_Expiry_Lot":
                    False,

                "Current_Lot_Level_Expiry_Exposure_Computable":
                    False,

                "Current_Expiring_Units_Computable":
                    False,

                "Current_Expiry_Value_Computable":
                    False,

                "Actual_Expired_Waste_Claim":
                    False,

                "No_Linked_History_Interpreted_As_No_Risk":
                    False,

                "Validated_DSS_Use":
                    (
                        "HISTORICAL_EXPIRY_EVIDENCE_"
                        "COVERAGE_QUALITY_AND_REVIEW_CONTEXT"
                    ),
            }
        ]
    )

    return governance


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_outputs(
    summary: pd.DataFrame,
    governance: pd.DataFrame,
    planning: pd.DataFrame,
    expiry: pd.DataFrame,
) -> None:

    if len(summary) != len(planning):

        raise ValueError(
            "DSS expiry summary row count "
            "does not match planning population."
        )

    if (
        summary["Medicine_Key"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "DSS expiry summary is not "
            "one row per Medicine_Key."
        )

    if (
        summary["Medicine_Key"]
        .isna()
        .any()
    ):

        raise ValueError(
            "DSS expiry summary contains "
            "null Medicine_Key."
        )

    expected_linked = int(
        expiry["Medicine_Key"]
        .nunique()
    )

    observed_linked = int(
        summary[
            "Has_Authoritative_Expiry_History"
        ].sum()
    )

    if observed_linked != expected_linked:

        raise ValueError(
            "Linked medicine count does not "
            "reconcile to authoritative "
            "expiry history."
        )

    if (
        summary[
            "Expiry_History_Record_Count"
        ].sum()
        !=
        len(expiry)
    ):

        raise ValueError(
            "Expiry history record counts "
            "do not reconcile."
        )

    no_history = (
        ~summary[
            "Has_Authoritative_Expiry_History"
        ]
    )

    history_count_columns = [
        "Expiry_History_Record_Count",
        "Expiry_History_Document_Count",
        "Usable_Lot_Record_Count",
        "Null_Or_Blank_Lot_Record_Count",
        "Zero_Placeholder_Lot_Record_Count",
        "Distinct_Usable_Lot_Value_Count",
        "Duplicate_Flagged_Record_Count",
        "Document_After_Expiration_Record_Count",
        "Net_Arithmetic_Mismatch_Record_Count",
        "Zero_Land_Cost_Record_Count",
        "Source_Review_Record_Count",
    ]

    if (
        summary.loc[
            no_history,
            history_count_columns,
        ]
        .ne(0)
        .any()
        .any()
    ):

        raise ValueError(
            "Medicine without linked expiry "
            "history has nonzero history counts."
        )

    if (
        summary.loc[
            no_history,
            "Expiry_Item_ID",
        ]
        .notna()
        .any()
    ):

        raise ValueError(
            "Medicine without expiry history "
            "contains an expiry Item_ID."
        )

    if (
        summary[
            "Planning_Stock_Linked_To_Expiry_Lot"
        ]
        .any()
    ):

        raise ValueError(
            "Planning stock must not be "
            "linked to expiry lots."
        )

    for column in [
        "Current_Lot_Level_Expiry_Exposure_Computable",
        "Current_Expiring_Units_Computable",
        "Current_Expiry_Value_Computable",
        "Actual_Expired_Waste_Claim",
    ]:

        if summary[column].any():

            raise ValueError(
                f"{column} must remain false."
            )

    if not (
        summary[
            "Linked_History_Does_Not_Imply_Current_Risk"
        ]
        .all()
    ):

        raise ValueError(
            "Historical-expiry interpretation "
            "guardrail failed."
        )

    if not (
        summary[
            "No_Linked_History_Does_Not_Imply_No_Risk"
        ]
        .all()
    ):

        raise ValueError(
            "No-history interpretation "
            "guardrail failed."
        )

    if len(governance) != 1:

        raise ValueError(
            "Expected one DSS expiry "
            "governance row."
        )

    if (
        governance.loc[
            0,
            "Planning_Medicine_Count",
        ]
        !=
        len(summary)
    ):

        raise ValueError(
            "Governance planning count "
            "does not reconcile."
        )

    if (
        governance.loc[
            0,
            "Medicines_With_Authoritative_Expiry_History",
        ]
        !=
        expected_linked
    ):

        raise ValueError(
            "Governance linked count "
            "does not reconcile."
        )


# ============================================================
# PERSISTENCE
# ============================================================

def write_outputs(
    summary: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:

    summary_parquet = (
        OUTPUT_DIR
        / "expiry_dss_medicine_summary.parquet"
    )

    summary_csv = (
        OUTPUT_DIR
        / "expiry_dss_medicine_summary.csv"
    )

    governance_csv = (
        OUTPUT_DIR
        / "expiry_dss_governance.csv"
    )

    summary.to_parquet(
        summary_parquet,
        index=False,
        engine="pyarrow",
    )

    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        governance_csv,
        index=False,
        encoding="utf-8-sig",
    )

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "expiry_dss_summary_df",
            summary,
        )

        connection.register(
            "expiry_dss_governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            expiry_dss_medicine_summary AS
            SELECT *
            FROM expiry_dss_summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            expiry_dss_governance AS
            SELECT *
            FROM expiry_dss_governance_df
            """
        )

    finally:

        connection.close()

    print()
    print("-" * 96)
    print("OUTPUTS")
    print("-" * 96)

    print(
        "DSS expiry medicine summary:",
        summary_parquet,
    )

    print(
        "DSS expiry governance:",
        governance_csv,
    )

    print(
        "DuckDB:",
        DATABASE_PATH,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 96)
    print("CDH INVENTORY DSS")
    print(
        "BUILD DSS-FACING EXPIRY "
        "MEDICINE SUMMARY"
    )
    print("=" * 96)
    print()

    (
        planning,
        expiry,
        review_queue,
        validation,
        source_governance,
    ) = load_inputs()

    validate_inputs(
        planning=planning,
        expiry=expiry,
        review_queue=review_queue,
        validation=validation,
        source_governance=(
            source_governance
        ),
    )

    summary = build_dss_summary(
        planning=planning,
        expiry=expiry,
        validation=validation,
        source_governance=(
            source_governance
        ),
    )

    governance = build_governance(
        summary=summary,
        expiry=expiry,
        review_queue=review_queue,
        validation=validation,
        source_governance=(
            source_governance
        ),
    )

    validate_outputs(
        summary=summary,
        governance=governance,
        planning=planning,
        expiry=expiry,
    )

    print(
        "Input, structural, and governance "
        "validation: PASS"
    )

    print()
    print("-" * 96)
    print("DSS EXPIRY COVERAGE")
    print("-" * 96)

    linked = int(
        summary[
            "Has_Authoritative_Expiry_History"
        ].sum()
    )

    print(
        "Planning medicines:",
        f"{len(summary):,}",
    )

    print(
        "With authoritative expiry history:",
        f"{linked:,}",
    )

    print(
        "Without linked expiry history:",
        f"{len(summary) - linked:,}",
    )

    print(
        "Coverage rate:",
        f"{linked / len(summary):.2%}",
    )

    print(
        "Authoritative historical "
        "expiry records:",
        f"{len(expiry):,}",
    )

    print(
        "Unresolved expiry Item_ID "
        "in separate review queue:",
        f"{len(review_queue):,}",
    )

    print()
    print("-" * 96)
    print("LOT DATA QUALITY")
    print("-" * 96)

    print(
        "Usable lot records:",
        f"{int(governance.loc[0, 'Usable_Lot_Record_Count']):,}",
    )

    print(
        "Blank/null lot records:",
        f"{int(governance.loc[0, 'Null_Or_Blank_Lot_Record_Count']):,}",
    )

    print(
        "Zero-placeholder lot records:",
        f"{int(governance.loc[0, 'Zero_Placeholder_Lot_Record_Count']):,}",
    )

    print()
    print("-" * 96)
    print("INTERPRETATION GUARDRAILS")
    print("-" * 96)

    print(
        "Planning stock linked to expiry lot: NO"
    )

    print(
        "Current expiring units computable: NO"
    )

    print(
        "Current expiry value computable: NO"
    )

    print(
        "Actual expired waste claim: NO"
    )

    print(
        "No linked history interpreted "
        "as no expiry risk: NO"
    )

    print(
        "Live forward expiry alert supported: "
        + (
            "YES"
            if bool(
                governance.loc[
                    0,
                    "Live_Forward_Expiry_Alert_Supported",
                ]
            )
            else "NO"
        )
    )

    write_outputs(
        summary=summary,
        governance=governance,
    )

    print()
    print("=" * 96)
    print(
        "DSS-FACING EXPIRY SUMMARY "
        "BUILD COMPLETE"
    )
    print("=" * 96)


if __name__ == "__main__":
    main()
