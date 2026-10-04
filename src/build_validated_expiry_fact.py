from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_EXPIRY_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "expiry"
)

VALIDATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "validated"
    / "expiry"
)

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "cdh_inventory.duckdb"
)

VALIDATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SOURCE CONTRACT
# ============================================================

REQUIRED_COLUMNS = [
    "Document_Date",
    "Document_No_",
    "Item_ID",
    "Item_Description",
    "Quantity",
    "Land_Cost",
    "Net_Amount",
    "Expiration_Date",
    "lotno",
    "Customer_Name",
    "Textbox46",
]

NET_AMOUNT_TOLERANCE = 0.01


# ============================================================
# HELPERS
# ============================================================

def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def clean_text_value(
    value,
):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value == "":
        return pd.NA

    return value


def normalize_description(
    value,
):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().upper()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    if value == "":
        return pd.NA

    return value


def item_prefix(
    value,
) -> str:

    if pd.isna(value):
        return "(NULL)"

    match = re.match(
        r"^[A-Za-z]+",
        str(value).strip(),
    )

    if match:
        return match.group(0).upper()

    return "(NO_ALPHA_PREFIX)"


def parse_report_metadata(
    path: Path,
) -> tuple[pd.Timestamp, pd.Timestamp, str]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="strict",
    ) as file:

        reader = csv.reader(file)

        records = []

        for _ in range(4):

            try:
                records.append(
                    next(reader)
                )

            except StopIteration as error:

                raise ValueError(
                    "Expiry source ended before "
                    "the expected report header."
                ) from error

    if len(records) != 4:

        raise ValueError(
            "Could not read the first four "
            "physical CSV records."
        )

    metadata_text = (
        records[1][0].strip()
        if records[1]
        else ""
    )

    match = re.search(
        r"Expiration Date From:\s*"
        r"(\d{1,2}/\d{1,2}/\d{4})"
        r"\s+To\s+"
        r"(\d{1,2}/\d{1,2}/\d{4})",
        metadata_text,
    )

    if not match:

        raise ValueError(
            "Could not parse expiry report "
            "date-range metadata."
        )

    report_from = pd.to_datetime(
        match.group(1),
        errors="raise",
    )

    report_to = pd.to_datetime(
        match.group(2),
        errors="raise",
    )

    physical_header = records[3]

    if physical_header != REQUIRED_COLUMNS:

        raise ValueError(
            "Unexpected expiry source header. "
            f"Observed: {physical_header}"
        )

    return (
        report_from,
        report_to,
        metadata_text,
    )


def get_source_file() -> Path:

    files = sorted(
        path
        for path in RAW_EXPIRY_DIR.glob("*.csv")
        if path.is_file()
    )

    if len(files) != 1:

        raise RuntimeError(
            "Expected exactly one raw expiry "
            f"CSV, found {len(files)}."
        )

    return files[0]


def load_medicine_dimension() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dimension = connection.execute(
            """
            SELECT DISTINCT
                Medicine_Key,
                Brand_Name
            FROM dim_medicine
            """
        ).df()

    finally:

        connection.close()

    if dimension.empty:

        raise RuntimeError(
            "dim_medicine is empty."
        )

    if (
        dimension["Medicine_Key"]
        .isna()
        .any()
    ):

        raise ValueError(
            "dim_medicine contains null "
            "Medicine_Key values."
        )

    if (
        dimension["Brand_Name"]
        .isna()
        .any()
    ):

        raise ValueError(
            "dim_medicine contains null "
            "Brand_Name values."
        )

    dimension[
        "Description_Normalized"
    ] = (
        dimension["Brand_Name"]
        .map(normalize_description)
    )

    ambiguity = (
        dimension.groupby(
            "Description_Normalized"
        )["Medicine_Key"]
        .nunique()
    )

    ambiguity = ambiguity[
        ambiguity > 1
    ]

    if not ambiguity.empty:

        raise ValueError(
            "Normalized Brand_Name is "
            "ambiguous across Medicine_Key."
        )

    return dimension


# ============================================================
# BUILD
# ============================================================

def build_expiry_outputs():

    source_path = get_source_file()

    (
        report_from,
        report_to,
        metadata_text,
    ) = parse_report_metadata(
        source_path
    )

    source_hash = file_sha256(
        source_path
    )

    raw = pd.read_csv(
        source_path,
        header=3,
        dtype="string",
        encoding="utf-8-sig",
        low_memory=False,
    )

    if list(raw.columns) != REQUIRED_COLUMNS:

        raise ValueError(
            "Parsed expiry columns do not "
            "match the expected source contract."
        )

    if raw.empty:

        raise ValueError(
            "Expiry source contains no data rows."
        )

    raw = raw.copy()

    raw["Source_Row_Number"] = (
        np.arange(
            len(raw),
            dtype=np.int64,
        )
        + 5
    )

    raw["Source_File"] = (
        source_path.name
    )

    raw["Source_File_SHA256"] = (
        source_hash
    )

    for column in [
        "Document_No_",
        "Item_ID",
        "Item_Description",
        "lotno",
        "Customer_Name",
        "Textbox46",
    ]:

        raw[column] = raw[
            column
        ].map(
            clean_text_value
        )

    required_nonblank = [
        "Document_Date",
        "Document_No_",
        "Item_ID",
        "Item_Description",
        "Quantity",
        "Land_Cost",
        "Net_Amount",
        "Expiration_Date",
        "Customer_Name",
        "Textbox46",
    ]

    if raw[
        required_nonblank
    ].isna().any().any():

        null_counts = (
            raw[
                required_nonblank
            ]
            .isna()
            .sum()
        )

        null_counts = null_counts[
            null_counts > 0
        ]

        raise ValueError(
            "Unexpected nulls in required "
            "expiry fields: "
            f"{null_counts.to_dict()}"
        )

    raw["Document_Date"] = (
        pd.to_datetime(
            raw["Document_Date"],
            errors="raise",
        )
    )

    raw["Expiration_Date"] = (
        pd.to_datetime(
            raw["Expiration_Date"],
            errors="raise",
        )
    )

    for column in [
        "Quantity",
        "Land_Cost",
        "Net_Amount",
    ]:

        raw[column] = pd.to_numeric(
            raw[column]
            .str.replace(
                ",",
                "",
                regex=False,
            ),
            errors="raise",
        )

    if (
        raw["Quantity"] <= 0
    ).any():

        raise ValueError(
            "Expiry Quantity must be positive."
        )

    if (
        raw["Land_Cost"] < 0
    ).any():

        raise ValueError(
            "Expiry Land_Cost cannot "
            "be negative."
        )

    if (
        raw["Net_Amount"] < 0
    ).any():

        raise ValueError(
            "Expiry Net_Amount cannot "
            "be negative."
        )

    identity_pairs = (
        raw[
            [
                "Item_ID",
                "Item_Description",
            ]
        ]
        .drop_duplicates()
    )

    id_description_counts = (
        identity_pairs.groupby(
            "Item_ID"
        )[
            "Item_Description"
        ]
        .nunique()
    )

    if (
        id_description_counts > 1
    ).any():

        raise ValueError(
            "At least one expiry Item_ID "
            "maps to multiple descriptions."
        )

    description_id_counts = (
        identity_pairs.groupby(
            "Item_Description"
        )[
            "Item_ID"
        ]
        .nunique()
    )

    if (
        description_id_counts > 1
    ).any():

        raise ValueError(
            "At least one expiry description "
            "maps to multiple Item_ID values."
        )

    dimension = (
        load_medicine_dimension()
    )

    dimension_lookup = (
        dimension
        .drop_duplicates(
            "Description_Normalized"
        )
        .set_index(
            "Description_Normalized"
        )
    )

    raw[
        "Description_Normalized"
    ] = (
        raw[
            "Item_Description"
        ]
        .map(
            normalize_description
        )
    )

    raw["Item_Prefix"] = (
        raw["Item_ID"]
        .map(item_prefix)
    )

    raw["Medicine_Key"] = (
        raw[
            "Description_Normalized"
        ]
        .map(
            dimension_lookup[
                "Medicine_Key"
            ]
        )
    )

    raw["Brand_Name"] = (
        raw[
            "Description_Normalized"
        ]
        .map(
            dimension_lookup[
                "Brand_Name"
            ]
        )
    )

    raw[
        "Flag_MED_Prefix"
    ] = (
        raw["Item_Prefix"]
        .eq("MED")
    )

    raw[
        "Flag_Exact_Medicine_Match"
    ] = (
        raw["Medicine_Key"]
        .notna()
    )

    raw[
        "Medicine_Domain_Record"
    ] = (
        raw[
            "Flag_MED_Prefix"
        ]
        |
        raw[
            "Flag_Exact_Medicine_Match"
        ]
    )

    raw["Mapping_Status"] = np.where(
        raw[
            "Flag_Exact_Medicine_Match"
        ],
        "AUTHORITATIVE_EXACT_MATCH",
        np.where(
            raw[
                "Flag_MED_Prefix"
            ],
            "UNRESOLVED_MED_PREFIX",
            "NOT_IN_MEDICINE_DOMAIN",
        ),
    )

    raw["Mapping_Method"] = np.where(
        raw[
            "Flag_Exact_Medicine_Match"
        ],
        (
            "EXACT_NORMALIZED_DESCRIPTION_"
            "TO_DIM_MEDICINE"
        ),
        "NONE",
    )

    raw["Medicine_Domain_Rule"] = (
        np.select(
            [
                (
                    raw[
                        "Flag_MED_Prefix"
                    ]
                    &
                    raw[
                        "Flag_Exact_Medicine_Match"
                    ]
                ),
                (
                    raw[
                        "Flag_MED_Prefix"
                    ]
                    &
                    ~raw[
                        "Flag_Exact_Medicine_Match"
                    ]
                ),
                (
                    ~raw[
                        "Flag_MED_Prefix"
                    ]
                    &
                    raw[
                        "Flag_Exact_Medicine_Match"
                    ]
                ),
            ],
            [
                "MED_PREFIX_AND_EXACT_MATCH",
                "MED_PREFIX_UNRESOLVED",
                "EXACT_MATCH_NON_MED_PREFIX",
            ],
            default=(
                "EXCLUDED_NON_MEDICINE"
            ),
        )
    )

    duplicate_subset = (
        REQUIRED_COLUMNS
    )

    raw[
        "Flag_Exact_Source_Duplicate"
    ] = (
        raw.duplicated(
            subset=duplicate_subset,
            keep=False,
        )
    )

    raw[
        "Flag_Document_After_Expiration"
    ] = (
        raw["Document_Date"]
        >
        raw["Expiration_Date"]
    )

    raw[
        "Computed_Quantity_x_Land_Cost"
    ] = (
        raw["Quantity"]
        *
        raw["Land_Cost"]
    )

    raw[
        "Net_Amount_Difference"
    ] = (
        raw["Net_Amount"]
        -
        raw[
            "Computed_Quantity_x_Land_Cost"
        ]
    )

    raw[
        "Flag_Net_Amount_Arithmetic_Mismatch"
    ] = (
        raw[
            "Net_Amount_Difference"
        ]
        .abs()
        >
        NET_AMOUNT_TOLERANCE
    )

    raw[
        "Flag_Zero_Land_Cost"
    ] = (
        raw["Land_Cost"]
        .eq(0)
    )

    raw["Expiration_Year"] = (
        raw["Expiration_Date"]
        .dt.year
        .astype("int64")
    )

    fact = (
        raw.loc[
            raw[
                "Medicine_Domain_Record"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    fact[
        "Data_Quality_Flag_Count"
    ] = (
        fact[
            [
                "Flag_Exact_Source_Duplicate",
                "Flag_Document_After_Expiration",
                "Flag_Net_Amount_Arithmetic_Mismatch",
                "Flag_Zero_Land_Cost",
            ]
        ]
        .astype("int64")
        .sum(axis=1)
    )

    fact["Data_Quality_Status"] = (
        np.where(
            fact[
                "Data_Quality_Flag_Count"
            ]
            > 0,
            "REVIEW",
            "PASS",
        )
    )

    crosswalk = (
        fact[
            [
                "Item_ID",
                "Item_Description",
                "Description_Normalized",
                "Item_Prefix",
                "Medicine_Key",
                "Brand_Name",
                "Mapping_Status",
                "Mapping_Method",
                "Medicine_Domain_Rule",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Mapping_Status",
                "Item_ID",
            ]
        )
        .reset_index(drop=True)
    )

    occurrences = (
        fact.groupby(
            [
                "Item_ID",
                "Item_Description",
            ],
            dropna=False,
        )
        .agg(
            Source_Row_Count=(
                "Source_Row_Number",
                "size",
            ),
            Earliest_Document_Date=(
                "Document_Date",
                "min",
            ),
            Latest_Document_Date=(
                "Document_Date",
                "max",
            ),
            Earliest_Expiration_Date=(
                "Expiration_Date",
                "min",
            ),
            Latest_Expiration_Date=(
                "Expiration_Date",
                "max",
            ),
        )
        .reset_index()
    )

    review_queue = (
        crosswalk.loc[
            crosswalk[
                "Mapping_Status"
            ]
            .eq(
                "UNRESOLVED_MED_PREFIX"
            )
        ]
        .merge(
            occurrences,
            on=[
                "Item_ID",
                "Item_Description",
            ],
            how="left",
            validate="one_to_one",
        )
        .sort_values("Item_ID")
        .reset_index(drop=True)
    )

    build_timestamp = datetime.now(
        timezone.utc
    ).replace(
        tzinfo=None
    )

    observed_min_expiry = (
        raw["Expiration_Date"].min()
    )

    observed_max_expiry = (
        raw["Expiration_Date"].max()
    )

    source_has_future_expiry = (
        observed_max_expiry.date()
        >=
        build_timestamp.date()
    )

    exact_mapped_ids = int(
        crosswalk.loc[
            crosswalk[
                "Mapping_Status"
            ]
            .eq(
                "AUTHORITATIVE_EXACT_MATCH"
            ),
            "Item_ID",
        ].nunique()
    )

    total_domain_ids = int(
        crosswalk["Item_ID"].nunique()
    )

    validation_summary = pd.DataFrame(
        [
            {
                "Source_File":
                    source_path.name,

                "Source_File_SHA256":
                    source_hash,

                "Raw_Row_Count":
                    int(len(raw)),

                "Raw_Unique_Item_ID":
                    int(
                        raw["Item_ID"]
                        .nunique()
                    ),

                "Medicine_Domain_Row_Count":
                    int(len(fact)),

                "Excluded_Nonmedicine_Row_Count":
                    int(
                        (
                            ~raw[
                                "Medicine_Domain_Record"
                            ]
                        ).sum()
                    ),

                "Medicine_Domain_Unique_Item_ID":
                    total_domain_ids,

                "Exact_Mapped_Row_Count":
                    int(
                        fact[
                            "Flag_Exact_Medicine_Match"
                        ].sum()
                    ),

                "Exact_Mapped_Unique_Item_ID":
                    exact_mapped_ids,

                "Exact_Mapped_Item_ID_Rate":
                    (
                        exact_mapped_ids
                        /
                        total_domain_ids
                    ),

                "Unresolved_Medicine_Item_ID":
                    int(len(review_queue)),

                "Medicine_Rows_Flagged_As_Duplicates":
                    int(
                        fact[
                            "Flag_Exact_Source_Duplicate"
                        ].sum()
                    ),

                "Medicine_Rows_Document_After_Expiration":
                    int(
                        fact[
                            "Flag_Document_After_Expiration"
                        ].sum()
                    ),

                "Medicine_Rows_Net_Arithmetic_Mismatch":
                    int(
                        fact[
                            "Flag_Net_Amount_Arithmetic_Mismatch"
                        ].sum()
                    ),

                "Medicine_Rows_Zero_Land_Cost":
                    int(
                        fact[
                            "Flag_Zero_Land_Cost"
                        ].sum()
                    ),

                "Report_Expiration_From":
                    report_from,

                "Report_Expiration_To":
                    report_to,

                "Observed_Expiration_Min":
                    observed_min_expiry,

                "Observed_Expiration_Max":
                    observed_max_expiry,

                "Source_Has_Future_Expiry_As_Of_Build":
                    bool(
                        source_has_future_expiry
                    ),
            }
        ]
    )

    governance = pd.DataFrame(
        [
            {
                "Build_Timestamp_UTC":
                    build_timestamp,

                "Pipeline_Name":
                    (
                        "build_validated_expiry_fact"
                    ),

                "Source_File":
                    source_path.name,

                "Source_File_SHA256":
                    source_hash,

                "Source_Report_Metadata":
                    metadata_text,

                "Source_Report_Expiration_From":
                    report_from,

                "Source_Report_Expiration_To":
                    report_to,

                "Medicine_Domain_Rule":
                    (
                        "MED_PREFIX_OR_EXACT_NORMALIZED_"
                        "DESCRIPTION_MATCH_TO_DIM_MEDICINE"
                    ),

                "Authoritative_Identity_Mapping_Rule":
                    (
                        "EXACT_NORMALIZED_DESCRIPTION_"
                        "MATCH_TO_DIM_MEDICINE_ONLY"
                    ),

                "Fuzzy_Mapping_Automatically_Accepted":
                    False,

                "Unresolved_MED_Items_Require_Review":
                    True,

                "Exact_Duplicates_Removed":
                    False,

                "Document_After_Expiration_Rows_Removed":
                    False,

                "Source_Quantity_Interpreted_As_Current_On_Hand":
                    False,

                "Source_Net_Amount_Interpreted_As_Current_Expiry_Exposure":
                    False,

                "Current_Lot_Level_Expiry_Exposure_Computable":
                    False,

                "Actual_Expired_Waste_Claim":
                    False,

                "Live_Forward_Expiry_Alert_Supported_As_Of_Build":
                    bool(
                        source_has_future_expiry
                    ),

                "Validated_Usage":
                    (
                        "HISTORICAL_EXPIRY_EVIDENCE_"
                        "IDENTITY_LINKAGE_AND_REVIEW"
                    ),
            }
        ]
    )

    validate_outputs(
        raw=raw,
        fact=fact,
        crosswalk=crosswalk,
        review_queue=review_queue,
        validation_summary=(
            validation_summary
        ),
        governance=governance,
        report_from=report_from,
        report_to=report_to,
    )

    return (
        source_path,
        raw,
        fact,
        crosswalk,
        review_queue,
        validation_summary,
        governance,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_outputs(
    raw: pd.DataFrame,
    fact: pd.DataFrame,
    crosswalk: pd.DataFrame,
    review_queue: pd.DataFrame,
    validation_summary: pd.DataFrame,
    governance: pd.DataFrame,
    report_from: pd.Timestamp,
    report_to: pd.Timestamp,
) -> None:

    if len(fact) > len(raw):

        raise ValueError(
            "Medicine-domain fact cannot "
            "exceed raw source row count."
        )

    if (
        len(fact)
        +
        int(
            (
                ~raw[
                    "Medicine_Domain_Record"
                ]
            ).sum()
        )
        !=
        len(raw)
    ):

        raise ValueError(
            "Medicine-domain partition does "
            "not reconcile to raw rows."
        )

    if (
        fact[
            "Medicine_Domain_Record"
        ]
        .eq(False)
        .any()
    ):

        raise ValueError(
            "Non-medicine row entered "
            "the expiry medicine fact."
        )

    unresolved = fact[
        "Mapping_Status"
    ].eq(
        "UNRESOLVED_MED_PREFIX"
    )

    if (
        fact.loc[
            unresolved,
            "Medicine_Key",
        ]
        .notna()
        .any()
    ):

        raise ValueError(
            "Unresolved mapping unexpectedly "
            "contains Medicine_Key."
        )

    exact = fact[
        "Mapping_Status"
    ].eq(
        "AUTHORITATIVE_EXACT_MATCH"
    )

    if (
        fact.loc[
            exact,
            "Medicine_Key",
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Authoritative exact mapping "
            "contains null Medicine_Key."
        )

    if (
        crosswalk["Item_ID"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Expiry crosswalk contains "
            "duplicate Item_ID."
        )

    expected_unresolved = int(
        crosswalk[
            "Mapping_Status"
        ]
        .eq(
            "UNRESOLVED_MED_PREFIX"
        )
        .sum()
    )

    if (
        len(review_queue)
        !=
        expected_unresolved
    ):

        raise ValueError(
            "Review queue does not reconcile "
            "to unresolved crosswalk items."
        )

    if (
        fact["Source_Row_Number"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Source_Row_Number is not unique."
        )

    if (
        fact["Source_Row_Number"].min()
        < 5
    ):

        raise ValueError(
            "Unexpected source row numbering."
        )

    observed_min = (
        raw["Expiration_Date"].min()
    )

    observed_max = (
        raw["Expiration_Date"].max()
    )

    if observed_min != report_from:

        raise ValueError(
            "Observed minimum expiration date "
            "does not match report metadata."
        )

    if observed_max != report_to:

        raise ValueError(
            "Observed maximum expiration date "
            "does not match report metadata."
        )

    if len(validation_summary) != 1:

        raise ValueError(
            "Expected one expiry validation "
            "summary row."
        )

    if len(governance) != 1:

        raise ValueError(
            "Expected one expiry governance row."
        )

    if bool(
        governance.loc[
            0,
            "Current_Lot_Level_Expiry_Exposure_Computable",
        ]
    ):

        raise ValueError(
            "Current lot-level exposure must "
            "remain explicitly unsupported."
        )

    if bool(
        governance.loc[
            0,
            "Actual_Expired_Waste_Claim",
        ]
    ):

        raise ValueError(
            "Actual expired waste claim must "
            "remain false."
        )


# ============================================================
# PERSISTENCE
# ============================================================

def write_outputs(
    fact: pd.DataFrame,
    crosswalk: pd.DataFrame,
    review_queue: pd.DataFrame,
    validation_summary: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:

    fact_parquet = (
        VALIDATED_DIR
        / "fact_expiry_medicine_records.parquet"
    )

    fact_csv = (
        VALIDATED_DIR
        / "fact_expiry_medicine_records.csv"
    )

    crosswalk_parquet = (
        VALIDATED_DIR
        / "expiry_medicine_identity_crosswalk.parquet"
    )

    crosswalk_csv = (
        VALIDATED_DIR
        / "expiry_medicine_identity_crosswalk.csv"
    )

    review_csv = (
        VALIDATED_DIR
        / "expiry_medicine_review_queue.csv"
    )

    validation_csv = (
        VALIDATED_DIR
        / "expiry_validation_summary.csv"
    )

    governance_csv = (
        VALIDATED_DIR
        / "expiry_source_governance.csv"
    )

    fact.to_parquet(
        fact_parquet,
        index=False,
        engine="pyarrow",
    )

    fact.to_csv(
        fact_csv,
        index=False,
        encoding="utf-8-sig",
    )

    crosswalk.to_parquet(
        crosswalk_parquet,
        index=False,
        engine="pyarrow",
    )

    crosswalk.to_csv(
        crosswalk_csv,
        index=False,
        encoding="utf-8-sig",
    )

    review_queue.to_csv(
        review_csv,
        index=False,
        encoding="utf-8-sig",
    )

    validation_summary.to_csv(
        validation_csv,
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
            "expiry_fact_df",
            fact,
        )

        connection.register(
            "expiry_crosswalk_df",
            crosswalk,
        )

        connection.register(
            "expiry_review_df",
            review_queue,
        )

        connection.register(
            "expiry_validation_df",
            validation_summary,
        )

        connection.register(
            "expiry_governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            fact_expiry_medicine_records AS
            SELECT *
            FROM expiry_fact_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            expiry_medicine_identity_crosswalk AS
            SELECT *
            FROM expiry_crosswalk_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            expiry_medicine_review_queue AS
            SELECT *
            FROM expiry_review_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            expiry_validation_summary AS
            SELECT *
            FROM expiry_validation_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            expiry_source_governance AS
            SELECT *
            FROM expiry_governance_df
            """
        )

    finally:

        connection.close()

    print()
    print("-" * 96)
    print("OUTPUTS")
    print("-" * 96)

    print(
        "Validated expiry fact:",
        fact_parquet,
    )

    print(
        "Identity crosswalk:",
        crosswalk_parquet,
    )

    print(
        "Review queue:",
        review_csv,
    )

    print(
        "Validation summary:",
        validation_csv,
    )

    print(
        "Governance:",
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
        "BUILD VALIDATED MEDICINE EXPIRY "
        "EVIDENCE FACT"
    )
    print("=" * 96)
    print()

    (
        source_path,
        raw,
        fact,
        crosswalk,
        review_queue,
        validation_summary,
        governance,
    ) = build_expiry_outputs()

    print(
        "Structural and governance validation: PASS"
    )

    print()
    print("-" * 96)
    print("SOURCE PROFILE")
    print("-" * 96)

    print(
        "Source:",
        source_path.name,
    )

    print(
        "Raw rows:",
        f"{len(raw):,}",
    )

    print(
        "Raw unique Item_ID:",
        f"{raw['Item_ID'].nunique():,}",
    )

    print()
    print("-" * 96)
    print("MEDICINE-DOMAIN EXPIRY FACT")
    print("-" * 96)

    print(
        "Medicine-domain rows:",
        f"{len(fact):,}",
    )

    print(
        "Unique medicine-domain Item_ID:",
        f"{crosswalk['Item_ID'].nunique():,}",
    )

    print(
        "Exact mapped rows:",
        f"{fact['Flag_Exact_Medicine_Match'].sum():,}",
    )

    print(
        "Exact mapped Item_ID:",
        f"{crosswalk['Medicine_Key'].notna().sum():,}",
    )

    print(
        "Unresolved MED Item_ID:",
        f"{len(review_queue):,}",
    )

    print(
        "Non-MED exact-match rows retained:",
        f"{((~fact['Flag_MED_Prefix']) & fact['Flag_Exact_Medicine_Match']).sum():,}",
    )

    print()
    print("-" * 96)
    print("DATA QUALITY")
    print("-" * 96)

    print(
        "Rows flagged as exact duplicates:",
        f"{fact['Flag_Exact_Source_Duplicate'].sum():,}",
    )

    print(
        "Rows where Document_Date > Expiration_Date:",
        f"{fact['Flag_Document_After_Expiration'].sum():,}",
    )

    print(
        "Rows where Net_Amount differs from "
        "Quantity x Land_Cost by > 0.01:",
        f"{fact['Flag_Net_Amount_Arithmetic_Mismatch'].sum():,}",
    )

    print(
        "Rows with zero Land_Cost:",
        f"{fact['Flag_Zero_Land_Cost'].sum():,}",
    )

    print()
    print("-" * 96)
    print("GOVERNANCE")
    print("-" * 96)

    print(
        "Current lot-level expiry exposure computable:",
        "NO",
    )

    print(
        "Source Quantity treated as current on-hand:",
        "NO",
    )

    print(
        "Source Net_Amount treated as current expiry exposure:",
        "NO",
    )

    print(
        "Actual expired waste claim:",
        "NO",
    )

    print(
        "Live forward expiry alert supported "
        "as of build:",
        (
            "YES"
            if bool(
                governance.loc[
                    0,
                    "Live_Forward_Expiry_Alert_Supported_As_Of_Build",
                ]
            )
            else "NO"
        ),
    )

    write_outputs(
        fact=fact,
        crosswalk=crosswalk,
        review_queue=review_queue,
        validation_summary=validation_summary,
        governance=governance,
    )

    print()
    print("=" * 96)
    print(
        "VALIDATED MEDICINE EXPIRY FACT "
        "BUILD COMPLETE"
    )
    print("=" * 96)

    print(
        "Raw source file was not modified."
    )


if __name__ == "__main__":
    main()
