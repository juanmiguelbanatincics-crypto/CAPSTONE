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
    / "planning_2026_inputs_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

PLANNING_VERSION = "PLANNING_2026_INPUTS_V1"

TARGET_YEAR = 2026
BASELINE_STOCK_YEAR = 2025
LATEST_COST_YEAR = 2025

EXPECTED_MEDICINES = 2733


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        forecast = connection.execute(
            """
            SELECT *
            FROM forecast_2026_candidate
            ORDER BY Medicine_Key
            """
        ).df()

        fact = connection.execute(
            """
            SELECT
                Medicine_Key,
                Year,
                Brand_Name,
                Generic_Name,
                Unit,
                Ending_Balance,
                Average_cost,
                Cost_Valid,
                Data_Quality_Status,
                Source_File
            FROM fact_inventory_annual
            WHERE Year BETWEEN 2021 AND 2025
            ORDER BY Medicine_Key, Year
            """
        ).df()

    finally:

        connection.close()

    return (
        forecast,
        fact,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(
    forecast: pd.DataFrame,
    fact: pd.DataFrame,
) -> None:

    if len(forecast) != EXPECTED_MEDICINES:

        raise ValueError(
            f"Expected {EXPECTED_MEDICINES:,} "
            f"2026 forecast rows; found "
            f"{len(forecast):,}"
        )

    if (
        forecast[
            "Medicine_Key"
        ]
        .nunique()
        != EXPECTED_MEDICINES
    ):

        raise ValueError(
            "Forecast Medicine_Key "
            "uniqueness failed."
        )

    if not (
        forecast[
            "Target_Year"
        ]
        .eq(
            TARGET_YEAR
        )
        .all()
    ):

        raise ValueError(
            "Forecast target year "
            "is not 2026."
        )

    if not (
        forecast[
            "Feature_Cutoff_Year"
        ]
        .eq(
            BASELINE_STOCK_YEAR
        )
        .all()
    ):

        raise ValueError(
            "Forecast feature cutoff "
            "is not 2025."
        )

    expected_fact_rows = (
        EXPECTED_MEDICINES
        * 5
    )

    if len(fact) != expected_fact_rows:

        raise ValueError(
            f"Expected {expected_fact_rows:,} "
            f"historical fact rows; "
            f"found {len(fact):,}"
        )

    duplicate_fact = int(
        fact.duplicated(
            subset=[
                "Medicine_Key",
                "Year",
            ]
        ).sum()
    )

    if duplicate_fact:

        raise ValueError(
            "Duplicate Medicine_Key-Year "
            "in historical fact."
        )


# ============================================================
# 2025 BASELINE STOCK
# ============================================================

def build_baseline_stock(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    stock = (
        fact.loc[
            fact[
                "Year"
            ]
            .eq(
                BASELINE_STOCK_YEAR
            ),
            [
                "Medicine_Key",
                "Ending_Balance",
                "Data_Quality_Status",
                "Source_File",
            ],
        ]
        .copy()
    )

    stock = stock.rename(
        columns={
            "Ending_Balance":
                "Source_Ending_Balance_2025",

            "Data_Quality_Status":
                "Stock_Source_Data_Quality_Status",

            "Source_File":
                "Stock_Source_File",
        }
    )

    missing = (
        stock[
            "Source_Ending_Balance_2025"
        ]
        .isna()
    )

    negative = (
        stock[
            "Source_Ending_Balance_2025"
        ]
        .lt(0)
        .fillna(False)
    )

    valid = (
        ~missing
        &
        ~negative
    )

    stock[
        "Planning_Stock_Status"
    ] = np.select(
        [
            missing,
            negative,
            valid,
        ],
        [
            "MISSING_2025_ENDING_BALANCE",
            "NEGATIVE_2025_ENDING_BALANCE",
            "VALID_2025_ENDING_BALANCE",
        ],
        default="UNKNOWN",
    )

    stock[
        "Planning_Stock_Valid"
    ] = valid

    stock[
        "Planning_Stock_Qty"
    ] = np.nan

    stock.loc[
        valid,
        "Planning_Stock_Qty",
    ] = (
        stock.loc[
            valid,
            "Source_Ending_Balance_2025",
        ]
    )

    stock[
        "Planning_Stock_Source_Year"
    ] = BASELINE_STOCK_YEAR

    stock[
        "Planning_Stock_Source_Type"
    ] = (
        "COMPLETED_ANNUAL_ENDING_BALANCE"
    )

    return stock


# ============================================================
# LATEST VALID HISTORICAL COST
# ============================================================

def build_planning_cost(
    fact: pd.DataFrame,
) -> pd.DataFrame:

    history = (
        fact.loc[
            fact[
                "Year"
            ]
            .le(
                LATEST_COST_YEAR
            )
        ]
        .copy()
    )

    valid_cost = (
        history[
            "Cost_Valid"
        ]
        .fillna(False)
        .astype(bool)
        &
        history[
            "Average_cost"
        ]
        .notna()
        &
        history[
            "Average_cost"
        ]
        .gt(0)
    )

    usable = (
        history.loc[
            valid_cost,
            [
                "Medicine_Key",
                "Year",
                "Average_cost",
                "Source_File",
            ],
        ]
        .sort_values(
            [
                "Medicine_Key",
                "Year",
            ]
        )
        .drop_duplicates(
            subset=[
                "Medicine_Key",
            ],
            keep="last",
        )
        .copy()
    )

    usable = usable.rename(
        columns={
            "Year":
                "Planning_Cost_Source_Year",

            "Average_cost":
                "Planning_Cost",

            "Source_File":
                "Planning_Cost_Source_File",
        }
    )

    all_medicines = (
        fact[
            [
                "Medicine_Key",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    cost = all_medicines.merge(
        usable,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    cost[
        "Planning_Cost_Available"
    ] = (
        cost[
            "Planning_Cost"
        ]
        .notna()
        &
        cost[
            "Planning_Cost"
        ]
        .gt(0)
    )

    cost[
        "Planning_Cost_Fallback_Used"
    ] = (
        cost[
            "Planning_Cost_Available"
        ]
        &
        cost[
            "Planning_Cost_Source_Year"
        ]
        .lt(
            LATEST_COST_YEAR
        )
    )

    cost[
        "Planning_Cost_Status"
    ] = np.select(
        [
            ~cost[
                "Planning_Cost_Available"
            ],

            cost[
                "Planning_Cost_Source_Year"
            ]
            .eq(
                LATEST_COST_YEAR
            ),

            cost[
                "Planning_Cost_Source_Year"
            ]
            .lt(
                LATEST_COST_YEAR
            ),
        ],
        [
            "UNAVAILABLE",
            "VALID_2025_COST",
            "HISTORICAL_VALID_COST_FALLBACK",
        ],
        default="UNKNOWN",
    )

    cost[
        "Planning_Cost_Age_Years"
    ] = np.nan

    available = (
        cost[
            "Planning_Cost_Available"
        ]
    )

    cost.loc[
        available,
        "Planning_Cost_Age_Years",
    ] = (
        LATEST_COST_YEAR
        -
        cost.loc[
            available,
            "Planning_Cost_Source_Year",
        ]
    )

    cost[
        "Planning_Cost_Source_Type"
    ] = np.where(
        cost[
            "Planning_Cost_Available"
        ],
        "LATEST_VALID_ANNUAL_AVERAGE_COST",
        "NONE",
    )

    return cost


# ============================================================
# BUILD PLANNING INPUT TABLE
# ============================================================

def build_planning_inputs(
    forecast: pd.DataFrame,
    stock: pd.DataFrame,
    cost: pd.DataFrame,
) -> pd.DataFrame:

    result = forecast.merge(
        stock,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        cost,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # REQUIREMENT SCENARIOS
    # --------------------------------------------------------

    result[
        "Requirement_Point"
    ] = (
        result[
            "Point_Forecast_2026"
        ]
    )

    result[
        "Requirement_P80"
    ] = (
        result[
            "P80"
        ]
    )

    result[
        "Requirement_P90"
    ] = (
        result[
            "P90"
        ]
    )

    # Do not silently force the quantile
    # forecasts above the incumbent point
    # forecast. Flag disagreements instead.

    result[
        "Flag_P80_Below_Point"
    ] = (
        result[
            "Requirement_P80"
        ]
        <
        result[
            "Requirement_Point"
        ]
    )

    result[
        "Flag_P90_Below_Point"
    ] = (
        result[
            "Requirement_P90"
        ]
        <
        result[
            "Requirement_Point"
        ]
    )

    # --------------------------------------------------------
    # NET REQUIREMENTS
    # --------------------------------------------------------

    stock_valid = (
        result[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    for scenario in [
        "Point",
        "P80",
        "P90",
    ]:

        requirement_column = (
            f"Requirement_{scenario}"
        )

        net_column = (
            f"Net_Requirement_{scenario}"
        )

        value_column = (
            f"Estimated_Procurement_Value_{scenario}"
        )

        result[
            net_column
        ] = np.nan

        result.loc[
            stock_valid,
            net_column,
        ] = np.maximum(
            0,
            (
                result.loc[
                    stock_valid,
                    requirement_column,
                ]
                -
                result.loc[
                    stock_valid,
                    "Planning_Stock_Qty",
                ]
            ),
        )

        # Value is calculated only when
        # BOTH stock and planning cost
        # are available.

        value_ready = (
            stock_valid
            &
            result[
                "Planning_Cost_Available"
            ]
            .fillna(False)
            .astype(bool)
        )

        result[
            value_column
        ] = np.nan

        result.loc[
            value_ready,
            value_column,
        ] = (
            result.loc[
                value_ready,
                net_column,
            ]
            *
            result.loc[
                value_ready,
                "Planning_Cost",
            ]
        )

    # --------------------------------------------------------
    # CORE OPTIMIZER READINESS
    # --------------------------------------------------------

    result[
        "Optimizer_Core_Input_Ready"
    ] = (
        result[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
        &
        result[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    stock_problem = (
        ~result[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    cost_problem = (
        ~result[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "Planning_Input_Status"
    ] = np.select(
        [
            (
                ~stock_problem
                &
                ~cost_problem
            ),

            (
                stock_problem
                &
                ~cost_problem
            ),

            (
                ~stock_problem
                &
                cost_problem
            ),

            (
                stock_problem
                &
                cost_problem
            ),
        ],
        [
            "READY",
            "STOCK_REVIEW_REQUIRED",
            "COST_REVIEW_REQUIRED",
            "STOCK_AND_COST_REVIEW_REQUIRED",
        ],
        default="UNKNOWN",
    )

    result[
        "Needs_Current_Stock_Override"
    ] = stock_problem

    result[
        "Needs_Planning_Cost_Override"
    ] = cost_problem

    # --------------------------------------------------------
    # GOVERNANCE
    # --------------------------------------------------------

    result[
        "Planning_Stock_Is_Current_Midyear"
    ] = False

    result[
        "Year_End_Pharmacy_Balance_Used"
    ] = False

    result[
        "Year_End_Implied_Cost_Used"
    ] = False

    result[
        "Last_Purchase_Price_Used"
    ] = False

    result[
        "Historical_ABC_Cost_Modified"
    ] = False

    result[
        "Planning_Version"
    ] = PLANNING_VERSION

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_planning_inputs(
    dataframe: pd.DataFrame,
) -> None:

    if len(
        dataframe
    ) != EXPECTED_MEDICINES:

        raise ValueError(
            f"Expected {EXPECTED_MEDICINES:,} "
            f"planning rows; "
            f"found {len(dataframe):,}"
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate Medicine_Key "
            "in planning inputs."
        )

    valid_stock = (
        dataframe[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    if (
        dataframe.loc[
            valid_stock,
            "Planning_Stock_Qty",
        ]
        .lt(0)
        .any()
    ):

        raise ValueError(
            "Negative valid planning stock."
        )

    valid_cost = (
        dataframe[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    if (
        dataframe.loc[
            valid_cost,
            "Planning_Cost",
        ]
        .le(0)
        .any()
    ):

        raise ValueError(
            "Non-positive available "
            "planning cost."
        )

    for scenario in [
        "Point",
        "P80",
        "P90",
    ]:

        requirement = (
            f"Requirement_{scenario}"
        )

        net = (
            f"Net_Requirement_{scenario}"
        )

        value = (
            f"Estimated_Procurement_Value_{scenario}"
        )

        if (
            dataframe[
                requirement
            ]
            .lt(0)
            .any()
        ):

            raise ValueError(
                f"Negative {requirement}."
            )

        if (
            dataframe.loc[
                valid_stock,
                net,
            ]
            .lt(0)
            .any()
        ):

            raise ValueError(
                f"Negative {net}."
            )

        ready = (
            valid_stock
            &
            valid_cost
        )

        if (
            dataframe.loc[
                ready,
                value,
            ]
            .lt(0)
            .any()
        ):

            raise ValueError(
                f"Negative {value}."
            )

    if (
        dataframe[
            "Year_End_Pharmacy_Balance_Used"
        ]
        .any()
    ):

        raise ValueError(
            "Year-end PHARMACY balance "
            "was unexpectedly used."
        )

    if (
        dataframe[
            "Last_Purchase_Price_Used"
        ]
        .any()
    ):

        raise ValueError(
            "Last Purchase Price was "
            "unexpectedly used."
        )


# ============================================================
# OVERALL READINESS SUMMARY
# ============================================================

def build_overall_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    row = {
        "Target_Year":
            TARGET_YEAR,

        "Medicine_Count":
            len(
                dataframe
            ),

        "Valid_2025_Stock_Count":
            int(
                dataframe[
                    "Planning_Stock_Status"
                ]
                .eq(
                    "VALID_2025_ENDING_BALANCE"
                )
                .sum()
            ),

        "Missing_2025_Stock_Count":
            int(
                dataframe[
                    "Planning_Stock_Status"
                ]
                .eq(
                    "MISSING_2025_ENDING_BALANCE"
                )
                .sum()
            ),

        "Negative_2025_Stock_Count":
            int(
                dataframe[
                    "Planning_Stock_Status"
                ]
                .eq(
                    "NEGATIVE_2025_ENDING_BALANCE"
                )
                .sum()
            ),

        "Valid_2025_Cost_Count":
            int(
                dataframe[
                    "Planning_Cost_Status"
                ]
                .eq(
                    "VALID_2025_COST"
                )
                .sum()
            ),

        "Historical_Cost_Fallback_Count":
            int(
                dataframe[
                    "Planning_Cost_Status"
                ]
                .eq(
                    "HISTORICAL_VALID_COST_FALLBACK"
                )
                .sum()
            ),

        "Planning_Cost_Unavailable_Count":
            int(
                dataframe[
                    "Planning_Cost_Status"
                ]
                .eq(
                    "UNAVAILABLE"
                )
                .sum()
            ),

        "Optimizer_Core_Ready_Count":
            int(
                dataframe[
                    "Optimizer_Core_Input_Ready"
                ]
                .sum()
            ),

        "Optimizer_Core_Review_Count":
            int(
                (
                    ~dataframe[
                        "Optimizer_Core_Input_Ready"
                    ]
                )
                .sum()
            ),

        "Point_Requirement_Total":
            float(
                dataframe[
                    "Requirement_Point"
                ]
                .sum()
            ),

        "P80_Requirement_Total":
            float(
                dataframe[
                    "Requirement_P80"
                ]
                .sum()
            ),

        "P90_Requirement_Total":
            float(
                dataframe[
                    "Requirement_P90"
                ]
                .sum()
            ),

        "P80_Below_Point_Count":
            int(
                dataframe[
                    "Flag_P80_Below_Point"
                ]
                .sum()
            ),

        "P90_Below_Point_Count":
            int(
                dataframe[
                    "Flag_P90_Below_Point"
                ]
                .sum()
            ),

        "Planning_Version":
            PLANNING_VERSION,
    }

    return pd.DataFrame(
        [
            row
        ]
    )


# ============================================================
# READINESS BY ABC
# ============================================================

def build_readiness_by_abc(
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
                    len(
                        group
                    ),

                "Optimizer_Ready_Count":
                    int(
                        group[
                            "Optimizer_Core_Input_Ready"
                        ]
                        .sum()
                    ),

                "Optimizer_Review_Count":
                    int(
                        (
                            ~group[
                                "Optimizer_Core_Input_Ready"
                            ]
                        )
                        .sum()
                    ),

                "Stock_Review_Count":
                    int(
                        (
                            ~group[
                                "Planning_Stock_Valid"
                            ]
                            .fillna(False)
                        )
                        .sum()
                    ),

                "Cost_Review_Count":
                    int(
                        (
                            ~group[
                                "Planning_Cost_Available"
                            ]
                            .fillna(False)
                        )
                        .sum()
                    ),

                "Historical_Cost_Fallback_Count":
                    int(
                        group[
                            "Planning_Cost_Fallback_Used"
                        ]
                        .fillna(False)
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
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# KNOWN NET REQUIREMENT / VALUE SUMMARY
# ============================================================

def build_known_value_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    stock_valid = (
        dataframe[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    fully_ready = (
        dataframe[
            "Optimizer_Core_Input_Ready"
        ]
        .fillna(False)
        .astype(bool)
    )

    for scenario in [
        "Point",
        "P80",
        "P90",
    ]:

        requirement_column = (
            f"Requirement_{scenario}"
        )

        net_column = (
            f"Net_Requirement_{scenario}"
        )

        value_column = (
            f"Estimated_Procurement_Value_{scenario}"
        )

        rows.append(
            {
                "Scenario":
                    scenario,

                "Gross_Requirement_Total":
                    float(
                        dataframe[
                            requirement_column
                        ]
                        .sum()
                    ),

                "Rows_With_Valid_Stock":
                    int(
                        stock_valid.sum()
                    ),

                "Known_Net_Requirement_Total":
                    float(
                        dataframe.loc[
                            stock_valid,
                            net_column,
                        ]
                        .sum()
                    ),

                "Rows_With_Stock_And_Cost":
                    int(
                        fully_ready.sum()
                    ),

                "Known_Estimated_Procurement_Value":
                    float(
                        dataframe.loc[
                            fully_ready,
                            value_column,
                        ]
                        .sum()
                    ),

                "Important_Limitation":
                    (
                        "Known totals exclude "
                        "records whose stock "
                        "and/or planning cost "
                        "requires review."
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "Planning_Version":
                    PLANNING_VERSION,

                "Target_Year":
                    2026,

                "Baseline_Stock_Year":
                    2025,

                "Planning_Stock_Policy":
                    (
                        "Use 2025 completed annual "
                        "Ending_Balance only when "
                        "observed and nonnegative."
                    ),

                "Missing_Stock_Policy":
                    (
                        "Do not convert missing "
                        "Ending_Balance to zero."
                    ),

                "Negative_Stock_Policy":
                    (
                        "Do not convert negative "
                        "Ending_Balance to zero; "
                        "require review/override."
                    ),

                "Planning_Cost_Policy":
                    (
                        "Use latest valid positive "
                        "annual Average_cost available "
                        "through 2025, retaining "
                        "source-year provenance."
                    ),

                "Historical_ACV_Modified":
                    False,

                "Last_Purchase_Price_Used":
                    False,

                "Year_End_Implied_Cost_Used":
                    False,

                "Year_End_Pharmacy_Balance_Used":
                    False,

                "P50_Used_As_Procurement_Requirement":
                    False,

                "Optimizer_Executed":
                    False,

                "Operational_Current_Snapshot_Rule":
                    (
                        "A validated current inventory "
                        "snapshot should supersede the "
                        "2025 closing balance when the "
                        "DSS is used after the start "
                        "of 2026."
                    ),
            }
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    inputs: pd.DataFrame,
    overall: pd.DataFrame,
    by_abc: pd.DataFrame,
    values: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "inputs_df",
            inputs,
        )

        connection.register(
            "overall_df",
            overall,
        )

        connection.register(
            "by_abc_df",
            by_abc,
        )

        connection.register(
            "values_df",
            values,
        )

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_inputs AS
            SELECT *
            FROM inputs_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_readiness_summary AS
            SELECT *
            FROM overall_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_readiness_by_abc AS
            SELECT *
            FROM by_abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_known_value_summary AS
            SELECT *
            FROM values_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_governance AS
            SELECT *
            FROM governance_df
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
    print("2026 PLANNING COST AND BASELINE STOCK READINESS")
    print("=" * 108)
    print()

    forecast, fact = (
        load_data()
    )

    validate_inputs(
        forecast,
        fact,
    )

    print(
        "Planning source input validation: PASS"
    )

    stock = (
        build_baseline_stock(
            fact
        )
    )

    cost = (
        build_planning_cost(
            fact
        )
    )

    planning = (
        build_planning_inputs(
            forecast,
            stock,
            cost,
        )
    )

    validate_planning_inputs(
        planning
    )

    print(
        "Planning input structural validation: PASS"
    )

    overall = (
        build_overall_summary(
            planning
        )
    )

    by_abc = (
        build_readiness_by_abc(
            planning
        )
    )

    values = (
        build_known_value_summary(
            planning
        )
    )

    governance = (
        build_governance()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    planning.to_parquet(
        OUTPUT_DIR
        / "planning_2026_inputs.parquet",
        index=False,
        engine="pyarrow",
    )

    planning.to_csv(
        OUTPUT_DIR
        / "planning_2026_inputs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        OUTPUT_DIR
        / "planning_2026_readiness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "planning_2026_readiness_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    values.to_csv(
        OUTPUT_DIR
        / "planning_2026_known_value_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "planning_2026_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        inputs=planning,
        overall=overall,
        by_abc=by_abc,
        values=values,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 108)
    print("OVERALL PLANNING READINESS")
    print("-" * 108)

    print(
        overall.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("READINESS BY 2025 ABC")
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
    print(
        "KNOWN NET REQUIREMENT / "
        "PROCUREMENT VALUE"
    )
    print("-" * 108)

    print(
        values.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 108)
    print("PLANNING GOVERNANCE")
    print("-" * 108)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 108)
    print("IMPORTANT")
    print("-" * 108)

    print(
        "2025 Ending_Balance is treated "
        "only as a start-of-2026 baseline "
        "planning stock when it is observed "
        "and nonnegative."
    )

    print(
        "Missing or negative balances are "
        "NOT converted to zero."
    )

    print(
        "Planning cost may use an earlier "
        "valid annual Average_cost when "
        "2025 cost is invalid, with the "
        "source year retained."
    )

    print(
        "This forward planning-cost fallback "
        "does NOT alter historical ABC or ACV."
    )

    print(
        "Last Purchase Price, year-end "
        "implied valuation cost, and PHARMACY "
        "year-end balances are NOT used "
        "automatically."
    )

    print(
        "No optimizer has been executed."
    )

    print()
    print("=" * 108)
    print(
        "2026 PLANNING INPUT READINESS "
        "BUILD COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()