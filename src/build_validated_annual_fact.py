from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pandera.pandas as pa

try:
    from .runtime_context import load_runtime_context_from_env
except ImportError:
    from runtime_context import load_runtime_context_from_env


# ============================================================
# GOVERNED RUNTIME CONTEXT
# ============================================================

RUNTIME_CONTEXT = load_runtime_context_from_env()

PROJECT_ROOT = RUNTIME_CONTEXT.project_root
RAW_ANNUAL_DIR = RUNTIME_CONTEXT.raw_annual_dir
VALIDATED_DIR = RUNTIME_CONTEXT.validated_annual_dir
DATABASE_PATH = RUNTIME_CONTEXT.database_path

EXPECTED_YEARS = (
    RUNTIME_CONTEXT.historical_years
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
# SOURCE FIELDS
# ============================================================

NUMERIC_COLUMNS = [
    "Beginning_Balance",
    "Qty_in_Deliveries",
    "Qty_in_Credit_notes",
    "Qty_in_Stock_Issuances",
    "Qty_in_Material_Productions",
    "Qty_in_Qty_Adjustments",
    "Qty_in_Total_In",
    "Qty_out_Purchase_Returns",
    "Qty_out_Sales",
    "Qty_out_Stock_Issuances",
    "Qty_out_Expense_Issuances",
    "Qty_out_Material_Productions",
    "Qty_out_Qty_Adjustments",
    "Qty_out_Total_Out",
    "Ending_Balance",
    "Average_cost",
]


# ============================================================
# HELPERS
# ============================================================

def infer_year(
    filename: str,
) -> int:

    match = re.search(
        r"(20\d{2})",
        filename,
    )

    if not match:
        raise ValueError(
            f"Could not infer year from "
            f"filename: {filename}"
        )

    return int(
        match.group(1)
    )


def read_csv_safely(
    path: Path,
) -> pd.DataFrame:

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:

        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

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


def clean_text(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            "",
            pd.NA,
        )
    )


def normalize_identity(
    value: str,
) -> str:

    value = str(
        value
    ).upper().strip()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def create_medicine_key(
    brand_name: str,
) -> str:

    normalized = (
        normalize_identity(
            brand_name
        )
    )

    digest = hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        "MED_"
        + digest[:16].upper()
    )


def percent(
    numerator: int | float,
    denominator: int | float,
) -> float | None:

    if denominator == 0:
        return None

    return round(
        numerator
        / denominator
        * 100,
        4,
    )


# ============================================================
# LOAD ONE ANNUAL FILE
# ============================================================

def load_annual_file(
    path: Path,
) -> pd.DataFrame:

    year = infer_year(
        path.name
    )

    source_hash = file_sha256(
        path
    )

    dataframe = read_csv_safely(
        path
    )

    dataframe.columns = [
        str(column).strip()
        for column
        in dataframe.columns
    ]

    # Preserve original CSV row location.
    # +2 accounts for:
    #   pandas index starting at 0
    #   CSV header occupying row 1
    dataframe[
        "Source_Row_Number"
    ] = (
        dataframe.index
        + 2
    )

    for column in [
        "Category1",
        "Generic_Name",
        "Brand_Name",
        "Unit",
    ]:

        dataframe[column] = (
            clean_text(
                dataframe[column]
            )
        )

    for column in (
        NUMERIC_COLUMNS
    ):

        dataframe[column] = (
            pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
        )

    medicines = (
        dataframe.loc[
            dataframe[
                "Category1"
            ]
            .fillna("")
            .str.upper()
            .eq("MEDICINES")
        ]
        .copy()
    )

    medicines[
        "Year"
    ] = year

    medicines[
        "Source_File"
    ] = path.name

    medicines[
        "Source_File_SHA256"
    ] = source_hash

    medicines[
        "Source_Period_Status"
    ] = (
        "COMPLETED_ANNUAL"
    )

    return medicines


# ============================================================
# BUILD FACT TABLE
# ============================================================

def build_fact_table() -> pd.DataFrame:

    files = sorted(
        RAW_ANNUAL_DIR.glob(
            "*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No annual inventory files found."
        )

    frames = [
        load_annual_file(
            path
        )
        for path in files
    ]

    fact = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # INTERNAL MEDICINE KEY
    # --------------------------------------------------------

    if fact[
        "Brand_Name"
    ].isna().any():

        raise ValueError(
            "Brand_Name contains missing values. "
            "Medicine_Key cannot be generated."
        )

    fact[
        "Medicine_Key"
    ] = (
        fact[
            "Brand_Name"
        ]
        .map(
            create_medicine_key
        )
    )

    # --------------------------------------------------------
    # AUTHORITATIVE ANNUAL ISSUANCE TARGET
    # --------------------------------------------------------

    fact[
        "Issuance_Source_Blank"
    ] = (
        fact[
            "Qty_out_Stock_Issuances"
        ]
        .isna()
    )

    fact[
        "Annual_Issuance"
    ] = (
        fact[
            "Qty_out_Stock_Issuances"
        ]
        .fillna(0)
        .astype(float)
    )

    if (
        fact[
            "Annual_Issuance"
        ]
        < 0
    ).any():

        raise ValueError(
            "Negative annual stock issuance "
            "detected. Build stopped."
        )

    fact[
        "Issuance_Active"
    ] = (
        fact[
            "Annual_Issuance"
        ]
        .gt(0)
    )

    # --------------------------------------------------------
    # COST SEMANTICS
    # --------------------------------------------------------

    cost = fact[
        "Average_cost"
    ]

    conditions = [
        cost.isna(),
        cost.eq(0),
        cost.lt(0),
        cost.gt(0),
    ]

    choices = [
        "MISSING",
        "ZERO",
        "NEGATIVE",
        "VALID",
    ]

    fact[
        "Cost_Status"
    ] = np.select(
        conditions,
        choices,
        default="UNKNOWN",
    )

    fact[
        "Cost_Valid"
    ] = (
        cost.notna()
        &
        cost.gt(0)
    )

    # --------------------------------------------------------
    # ANNUAL CONSUMPTION VALUE
    # --------------------------------------------------------

    fact[
        "Annual_Consumption_Value"
    ] = np.nan

    # No recorded issuance means historical
    # consumption value is zero even if a
    # unit cost is not present.
    zero_issuance = (
        fact[
            "Annual_Issuance"
        ]
        .eq(0)
    )

    fact.loc[
        zero_issuance,
        "Annual_Consumption_Value",
    ] = 0.0

    valid_acv = (
        fact[
            "Annual_Issuance"
        ]
        .gt(0)
        &
        fact[
            "Cost_Valid"
        ]
    )

    fact.loc[
        valid_acv,
        "Annual_Consumption_Value",
    ] = (
        fact.loc[
            valid_acv,
            "Annual_Issuance",
        ]
        *
        fact.loc[
            valid_acv,
            "Average_cost",
        ]
    )

    fact[
        "ACV_Computable"
    ] = (
        zero_issuance
        |
        valid_acv
    )

    fact[
        "ACV_Status"
    ] = np.select(
        [
            zero_issuance,
            valid_acv,
        ],
        [
            "ZERO_ISSUANCE",
            "VALID",
        ],
        default=(
            "COST_INVALID"
        ),
    )

    # --------------------------------------------------------
    # BALANCE AVAILABILITY
    # --------------------------------------------------------

    beginning_present = (
        fact[
            "Beginning_Balance"
        ].notna()
    )

    ending_present = (
        fact[
            "Ending_Balance"
        ].notna()
    )

    fact[
        "Balance_Availability"
    ] = np.select(
        [
            (
                beginning_present
                &
                ending_present
            ),
            (
                beginning_present
                &
                ~ending_present
            ),
            (
                ~beginning_present
                &
                ending_present
            ),
        ],
        [
            "BOTH_PRESENT",
            "BEGINNING_ONLY",
            "ENDING_ONLY",
        ],
        default="BOTH_MISSING",
    )

    # --------------------------------------------------------
    # STOCK-FLOW VALIDATION
    # --------------------------------------------------------

    required_flow_columns = [
        "Beginning_Balance",
        "Qty_in_Total_In",
        "Qty_out_Total_Out",
        "Ending_Balance",
    ]

    fact[
        "Stock_Flow_Testable"
    ] = (
        fact[
            required_flow_columns
        ]
        .notna()
        .all(
            axis=1
        )
    )

    fact[
        "Stock_Flow_Difference"
    ] = np.nan

    testable = (
        fact[
            "Stock_Flow_Testable"
        ]
    )

    expected_ending = (
        fact.loc[
            testable,
            "Beginning_Balance",
        ]
        +
        fact.loc[
            testable,
            "Qty_in_Total_In",
        ]
        -
        fact.loc[
            testable,
            "Qty_out_Total_Out",
        ]
    )

    fact.loc[
        testable,
        "Stock_Flow_Difference",
    ] = (
        fact.loc[
            testable,
            "Ending_Balance",
        ]
        -
        expected_ending
    )

    fact[
        "Stock_Flow_Reconciled"
    ] = False

    fact.loc[
        testable,
        "Stock_Flow_Reconciled",
    ] = np.isclose(
        fact.loc[
            testable,
            "Stock_Flow_Difference",
        ],
        0,
        atol=0.0001,
        rtol=0,
    )

    fact[
        "Stock_Flow_Status"
    ] = np.select(
        [
            ~fact[
                "Stock_Flow_Testable"
            ],
            fact[
                "Stock_Flow_Reconciled"
            ],
        ],
        [
            "NOT_TESTABLE",
            "PASS",
        ],
        default="FAIL",
    )

    # --------------------------------------------------------
    # DATA-QUALITY FLAGS
    # --------------------------------------------------------

    fact[
        "Flag_Missing_Generic"
    ] = (
        fact[
            "Generic_Name"
        ]
        .isna()
    )

    fact[
        "Flag_Negative_Beginning_Balance"
    ] = (
        fact[
            "Beginning_Balance"
        ]
        .lt(0)
        .fillna(False)
    )

    fact[
        "Flag_Negative_Ending_Balance"
    ] = (
        fact[
            "Ending_Balance"
        ]
        .lt(0)
        .fillna(False)
    )

    fact[
        "Flag_Invalid_Cost_Positive_Issuance"
    ] = (
        fact[
            "Annual_Issuance"
        ]
        .gt(0)
        &
        ~fact[
            "Cost_Valid"
        ]
    )

    fact[
        "Flag_Stock_Flow_Mismatch"
    ] = (
        fact[
            "Stock_Flow_Status"
        ]
        .eq("FAIL")
    )

    quality_flag_columns = [
        "Flag_Missing_Generic",
        "Flag_Negative_Beginning_Balance",
        "Flag_Negative_Ending_Balance",
        "Flag_Invalid_Cost_Positive_Issuance",
        "Flag_Stock_Flow_Mismatch",
    ]

    fact[
        "Data_Quality_Flag_Count"
    ] = (
        fact[
            quality_flag_columns
        ]
        .astype(int)
        .sum(
            axis=1
        )
    )

    fact[
        "Data_Quality_Status"
    ] = np.where(
        fact[
            "Data_Quality_Flag_Count"
        ]
        .eq(0),
        "PASS",
        "REVIEW",
    )

    # --------------------------------------------------------
    # MODELING TARGET LABEL
    # --------------------------------------------------------

    fact[
        "Target_Definition"
    ] = (
        "OBSERVED_ANNUAL_STOCK_ISSUANCE_PROXY"
    )

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

    preferred_columns = [
        "Medicine_Key",
        "Year",
        "Brand_Name",
        "Generic_Name",
        "Unit",

        "Annual_Issuance",
        "Issuance_Active",
        "Issuance_Source_Blank",

        "Beginning_Balance",
        "Qty_in_Deliveries",
        "Qty_in_Total_In",

        "Qty_out_Stock_Issuances",
        "Qty_out_Expense_Issuances",
        "Qty_out_Qty_Adjustments",
        "Qty_out_Total_Out",

        "Ending_Balance",

        "Average_cost",
        "Cost_Status",
        "Cost_Valid",

        "Annual_Consumption_Value",
        "ACV_Computable",
        "ACV_Status",

        "Balance_Availability",

        "Stock_Flow_Testable",
        "Stock_Flow_Difference",
        "Stock_Flow_Reconciled",
        "Stock_Flow_Status",

        "Data_Quality_Status",
        "Data_Quality_Flag_Count",

        "Flag_Missing_Generic",
        "Flag_Negative_Beginning_Balance",
        "Flag_Negative_Ending_Balance",
        "Flag_Invalid_Cost_Positive_Issuance",
        "Flag_Stock_Flow_Mismatch",

        "Target_Definition",

        "Source_Period_Status",
        "Source_File",
        "Source_File_SHA256",
        "Source_Row_Number",
    ]

    fact = fact[
        preferred_columns
    ].copy()

    return fact


# ============================================================
# BUILD MEDICINE DIMENSION
# ============================================================

def build_medicine_dimension(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    dimension = (
        fact.groupby(
            "Medicine_Key",
            as_index=False,
        )
        .agg(
            Brand_Name=(
                "Brand_Name",
                "first",
            ),
            Generic_Name=(
                "Generic_Name",
                "first",
            ),
            Unit=(
                "Unit",
                "first",
            ),
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

    dimension[
        "Identity_Source"
    ] = "ANNUAL_BRAND_NAME"

    # Retained as frozen baseline audit evidence.
    dimension[
        "Identity_Stable_2021_2025"
    ] = (
        dimension[
            "First_Observed_Year"
        ]
        .le(2021)
        &
        dimension[
            "Last_Observed_Year"
        ]
        .ge(2025)
        &
        dimension[
            "Years_Observed"
        ]
        .ge(5)
    )

    return dimension


# ============================================================
# PANDERA VALIDATION
# ============================================================

def validate_fact_schema(
    fact: pd.DataFrame,
) -> None:

    schema = pa.DataFrameSchema(
        {
            "Medicine_Key":
                pa.Column(
                    str,
                    nullable=False,
                ),

            "Year":
                pa.Column(
                    int,
                    checks=pa.Check.isin(
                        EXPECTED_YEARS
                    ),
                    nullable=False,
                ),

            "Brand_Name":
                pa.Column(
                    str,
                    nullable=False,
                ),

            "Annual_Issuance":
                pa.Column(
                    float,
                    checks=pa.Check.ge(0),
                    nullable=False,
                ),

            "Issuance_Active":
                pa.Column(
                    bool,
                    nullable=False,
                ),

            "Cost_Valid":
                pa.Column(
                    bool,
                    nullable=False,
                ),

            "ACV_Computable":
                pa.Column(
                    bool,
                    nullable=False,
                ),

            "Stock_Flow_Testable":
                pa.Column(
                    bool,
                    nullable=False,
                ),

            "Stock_Flow_Reconciled":
                pa.Column(
                    bool,
                    nullable=False,
                ),

            "Data_Quality_Status":
                pa.Column(
                    str,
                    checks=pa.Check.isin(
                        [
                            "PASS",
                            "REVIEW",
                        ]
                    ),
                    nullable=False,
                ),

            "Source_File":
                pa.Column(
                    str,
                    nullable=False,
                ),
        },
        strict=False,
        coerce=True,
    )

    schema.validate(
        fact,
        lazy=True,
    )


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_structure(
    fact: pd.DataFrame,
    medicine_dimension: pd.DataFrame,
) -> None:

    actual_years = sorted(
        fact[
            "Year"
        ]
        .unique()
        .tolist()
    )

    if (
        actual_years
        != EXPECTED_YEARS
    ):
        raise ValueError(
            "Completed-year sequence validation failed. "
            f"Expected {EXPECTED_YEARS}; "
            f"found {actual_years}."
        )

    duplicate_key_year = int(
        fact.duplicated(
            subset=[
                "Medicine_Key",
                "Year",
            ]
        ).sum()
    )

    if duplicate_key_year != 0:
        raise ValueError(
            "Duplicate Medicine_Key-Year "
            f"records found: "
            f"{duplicate_key_year}"
        )

    collision_check = int(
        medicine_dimension[
            "Medicine_Key"
        ]
        .duplicated()
        .sum()
    )

    if collision_check:
        raise ValueError(
            "Medicine_Key hash collision "
            "detected."
        )

    # --------------------------------------------------------
    # FROZEN RESEARCH BASELINE REGRESSION CONTRACT
    # --------------------------------------------------------

    if RUNTIME_CONTEXT.is_baseline:

        if EXPECTED_YEARS != [
            2021,
            2022,
            2023,
            2024,
            2025,
        ]:
            raise ValueError(
                "BASELINE mode must represent "
                "the frozen 2021-2025 research period."
            )

        if len(fact) != 13665:
            raise ValueError(
                "Expected 13,665 medicine-year "
                f"rows, found {len(fact):,}"
            )

        if (
            fact[
                "Medicine_Key"
            ]
            .nunique()
            != 2733
        ):
            raise ValueError(
                "Expected 2,733 unique medicines "
                "for the frozen baseline."
            )

        medicine_counts = (
            fact.groupby(
                "Medicine_Key"
            )[
                "Year"
            ]
            .nunique()
        )

        if not (
            medicine_counts
            .eq(5)
            .all()
        ):
            raise ValueError(
                "Frozen baseline regression failed: "
                "not every medicine occurs in all "
                "five completed years."
            )

        return

    # --------------------------------------------------------
    # GOVERNED STAGING / TEST / CURRENT VALIDATION
    # --------------------------------------------------------

    if fact.empty:
        raise ValueError(
            "No validated annual medicine records "
            "were produced."
        )

    year_counts = (
        fact.groupby(
            "Year"
        )[
            "Medicine_Key"
        ]
        .nunique()
    )

    if (
        year_counts
        .le(0)
        .any()
    ):
        raise ValueError(
            "At least one configured completed year "
            "contains no medicine records."
        )

    # A future formulary is not required to remain a
    # permanently balanced 2,733-medicine panel.
    # New, continuing, and no-longer-observed medicines
    # are preserved as observed and reviewed downstream.

# ============================================================
# VALIDATION SUMMARY
# ============================================================

def build_validation_summary(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        fact.groupby(
            "Year"
        )
    ):

        active = int(
            year_df[
                "Issuance_Active"
            ]
            .sum()
        )

        invalid_cost_positive = int(
            year_df[
                "Flag_Invalid_Cost_Positive_Issuance"
            ]
            .sum()
        )

        stock_flow_testable = int(
            year_df[
                "Stock_Flow_Testable"
            ]
            .sum()
        )

        stock_flow_pass = int(
            (
                year_df[
                    "Stock_Flow_Status"
                ]
                == "PASS"
            )
            .sum()
        )

        rows.append(
            {
                "Year":
                    int(year),

                "Medicine_Rows":
                    len(year_df),

                "Active_Issuance_Medicines":
                    active,

                "Annual_Issuance_Total":
                    float(
                        year_df[
                            "Annual_Issuance"
                        ]
                        .sum()
                    ),

                "Issuance_Source_Blanks":
                    int(
                        year_df[
                            "Issuance_Source_Blank"
                        ]
                        .sum()
                    ),

                "Valid_Cost_Records":
                    int(
                        year_df[
                            "Cost_Valid"
                        ]
                        .sum()
                    ),

                "Positive_Issuance_Invalid_Cost":
                    invalid_cost_positive,

                "ACV_Computable_Records":
                    int(
                        year_df[
                            "ACV_Computable"
                        ]
                        .sum()
                    ),

                "Stock_Flow_Testable":
                    stock_flow_testable,

                "Stock_Flow_PASS":
                    stock_flow_pass,

                "Stock_Flow_PASS_Pct":
                    percent(
                        stock_flow_pass,
                        stock_flow_testable,
                    ),

                "Data_Quality_REVIEW":
                    int(
                        (
                            year_df[
                                "Data_Quality_Status"
                            ]
                            == "REVIEW"
                        )
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WRITE DUCKDB
# ============================================================

def write_duckdb(
    fact: pd.DataFrame,
    medicine_dimension: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(
            DATABASE_PATH
        )
    )

    try:

        connection.register(
            "fact_dataframe",
            fact,
        )

        connection.register(
            "medicine_dimension_dataframe",
            medicine_dimension,
        )

        connection.register(
            "validation_summary_dataframe",
            validation_summary,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            fact_inventory_annual AS
            SELECT *
            FROM fact_dataframe
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            dim_medicine AS
            SELECT *
            FROM medicine_dimension_dataframe
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            annual_validation_summary AS
            SELECT *
            FROM validation_summary_dataframe
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            etl_run_audit (
                run_timestamp TIMESTAMP,
                pipeline_name VARCHAR,
                fact_row_count BIGINT,
                medicine_count BIGINT,
                year_min INTEGER,
                year_max INTEGER,
                issuance_total DOUBLE,
                notes VARCHAR
            )
            """
        )

        run_timestamp = datetime.now(
            timezone.utc
        ).replace(
            tzinfo=None
        )

        connection.execute(
            """
            INSERT INTO etl_run_audit
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_timestamp,
                "build_validated_annual_fact",
                len(fact),
                int(
                    fact[
                        "Medicine_Key"
                    ].nunique()
                ),
                int(
                    fact[
                        "Year"
                    ].min()
                ),
                int(
                    fact[
                        "Year"
                    ].max()
                ),
                float(
                    fact[
                        "Annual_Issuance"
                    ]
                    .sum()
                ),
                (
                    f"Completed annual MEDICINES "
                    f"inventory, {EXPECTED_YEARS[0]}-"
                    f"{EXPECTED_YEARS[-1]}. "
                    "Blank stock issuance "
                    "standardized to zero recorded "
                    "issuance. Annual Average_cost "
                    "not replaced by year-end data."
                ),
            ],
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 88)
    print("CDH INVENTORY DSS")
    print("BUILD VALIDATED ANNUAL INVENTORY FACT TABLE")
    print("=" * 88)
    print()

    fact = build_fact_table()

    medicine_dimension = (
        build_medicine_dimension(
            fact
        )
    )

    print(
        "Running Pandera schema validation..."
    )

    validate_fact_schema(
        fact
    )

    print(
        "Pandera validation: PASS"
    )

    print(
        "Running structural validation..."
    )

    validate_structure(
        fact,
        medicine_dimension,
    )

    print(
        "Structural validation: PASS"
    )

    validation_summary = (
        build_validation_summary(
            fact
        )
    )

    # --------------------------------------------------------
    # SAVE PARQUET
    # --------------------------------------------------------

    fact_parquet = (
        VALIDATED_DIR
        / "fact_inventory_annual.parquet"
    )

    dimension_parquet = (
        VALIDATED_DIR
        / "dim_medicine.parquet"
    )

    fact.to_parquet(
        fact_parquet,
        index=False,
        engine="pyarrow",
    )

    medicine_dimension.to_parquet(
        dimension_parquet,
        index=False,
        engine="pyarrow",
    )

    # --------------------------------------------------------
    # SAVE CSV INSPECTION COPIES
    # --------------------------------------------------------

    fact_csv = (
        VALIDATED_DIR
        / "fact_inventory_annual.csv"
    )

    validation_csv = (
        VALIDATED_DIR
        / "annual_validation_summary.csv"
    )

    fact.to_csv(
        fact_csv,
        index=False,
        encoding="utf-8-sig",
    )

    validation_summary.to_csv(
        validation_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # DUCKDB
    # --------------------------------------------------------

    write_duckdb(
        fact,
        medicine_dimension,
        validation_summary,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 88)
    print("VALIDATED FACT TABLE")
    print("-" * 88)

    print(
        "Medicine-year records:",
        f"{len(fact):,}",
    )

    print(
        "Unique medicines:",
        f"{fact['Medicine_Key'].nunique():,}",
    )

    print(
        "Years:",
        sorted(
            fact[
                "Year"
            ]
            .unique()
            .tolist()
        ),
    )

    print(
        "Total annual issuance "
        f"across {EXPECTED_YEARS[0]}-"
        f"{EXPECTED_YEARS[-1]}:",
        f"{fact['Annual_Issuance'].sum():,.2f}",
    )

    print()
    print(
        "Duplicate Medicine_Key-Year:",
        int(
            fact.duplicated(
                [
                    "Medicine_Key",
                    "Year",
                ]
            ).sum()
        ),
    )

    print()
    print("-" * 88)
    print("YEAR-BY-YEAR VALIDATION SUMMARY")
    print("-" * 88)

    print(
        validation_summary
        .to_string(
            index=False
        )
    )

    print()
    print("-" * 88)
    print("COST POLICY")
    print("-" * 88)

    positive_invalid_cost = int(
        fact[
            "Flag_Invalid_Cost_Positive_Issuance"
        ]
        .sum()
    )

    print(
        "Positive-issuance medicine-years "
        "with invalid annual cost:",
        f"{positive_invalid_cost:,}",
    )

    print(
        "These records retain missing/"
        "invalid historical ACV."
    )

    print(
        "Year-end implied valuation cost "
        "was NOT used as a replacement."
    )

    print(
        "Last Purchase Price was NOT "
        "used as Average Cost."
    )

    print()
    print("-" * 88)
    print("OUTPUTS")
    print("-" * 88)

    print(
        "Fact Parquet:"
    )
    print(
        f"  {fact_parquet}"
    )

    print(
        "Medicine Dimension:"
    )
    print(
        f"  {dimension_parquet}"
    )

    print(
        "Inspection CSV:"
    )
    print(
        f"  {fact_csv}"
    )

    print(
        "Validation Summary:"
    )
    print(
        f"  {validation_csv}"
    )

    print(
        "DuckDB:"
    )
    print(
        f"  {DATABASE_PATH}"
    )

    print()
    print("=" * 88)
    print(
        "VALIDATED ANNUAL FACT BUILD COMPLETE"
    )
    print("=" * 88)

    print(
        "Raw source files were not modified."
    )


if __name__ == "__main__":
    main()