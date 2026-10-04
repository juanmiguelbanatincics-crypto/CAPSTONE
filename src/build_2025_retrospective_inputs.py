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
    / "retrospective"
    / "retrospective_2025_inputs_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

RETROSPECTIVE_VERSION = (
    "RETROSPECTIVE_2025_INPUTS_V1"
)

TARGET_YEAR = 2025
PLANNING_ORIGIN_YEAR = 2024

EXPECTED_MEDICINES = 2733

PRIORITY_CLASSES = [
    "A",
    "B",
    "C",
]

SCENARIOS = [
    "POINT",
    "P80_FLOORED_CEIL",
    "P90_FLOORED_CEIL",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        supervised = connection.execute(
            """
            SELECT *
            FROM forecasting_supervised
            WHERE Target_Year = 2025
            ORDER BY Medicine_Key
            """
        ).df()

        quantiles = connection.execute(
            """
            SELECT
                Medicine_Key,
                P50,
                P80,
                P90,
                Raw_P50,
                Raw_P80,
                Raw_P90,
                Quantile_Adjustment_Applied
            FROM forecast_quantile_predictions
            WHERE Fold_Name = 'BACKTEST_2025'
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
            WHERE Year BETWEEN 2021 AND 2024
            ORDER BY Medicine_Key, Year
            """
        ).df()

    finally:

        connection.close()

    return (
        supervised,
        quantiles,
        fact,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_sources(
    supervised: pd.DataFrame,
    quantiles: pd.DataFrame,
    fact: pd.DataFrame,
):

    if len(supervised) != EXPECTED_MEDICINES:

        raise ValueError(
            f"Expected {EXPECTED_MEDICINES:,} "
            f"2025 supervised rows; found "
            f"{len(supervised):,}"
        )

    if len(quantiles) != EXPECTED_MEDICINES:

        raise ValueError(
            f"Expected {EXPECTED_MEDICINES:,} "
            f"2025 quantile rows; found "
            f"{len(quantiles):,}"
        )

    expected_fact_rows = (
        EXPECTED_MEDICINES
        * 4
    )

    if len(fact) != expected_fact_rows:

        raise ValueError(
            f"Expected {expected_fact_rows:,} "
            f"historical rows through 2024; "
            f"found {len(fact):,}"
        )

    if supervised[
        "Medicine_Key"
    ].duplicated().any():

        raise ValueError(
            "Duplicate supervised Medicine_Key."
        )

    if quantiles[
        "Medicine_Key"
    ].duplicated().any():

        raise ValueError(
            "Duplicate quantile Medicine_Key."
        )

    if not supervised[
        "Feature_Cutoff_Year"
    ].eq(
        PLANNING_ORIGIN_YEAR
    ).all():

        raise ValueError(
            "2025 retrospective cutoff "
            "is not 2024."
        )

    if not supervised[
        "Max_History_Year_Used"
    ].eq(
        PLANNING_ORIGIN_YEAR
    ).all():

        raise ValueError(
            "Future information detected "
            "in 2025 forecast features."
        )

    if fact[
        "Year"
    ].gt(
        PLANNING_ORIGIN_YEAR
    ).any():

        raise ValueError(
            "Fact input contains data "
            "after the planning origin."
        )


# ============================================================
# BASELINE STOCK — END OF 2024
# ============================================================

def build_stock(
    fact: pd.DataFrame,
):

    stock = fact.loc[
        fact[
            "Year"
        ].eq(
            PLANNING_ORIGIN_YEAR
        ),
        [
            "Medicine_Key",
            "Ending_Balance",
            "Data_Quality_Status",
            "Source_File",
        ],
    ].copy()

    stock = stock.rename(
        columns={
            "Ending_Balance":
                "Source_Ending_Balance_2024",

            "Data_Quality_Status":
                "Stock_Source_Data_Quality_Status",

            "Source_File":
                "Stock_Source_File",
        }
    )

    missing = (
        stock[
            "Source_Ending_Balance_2024"
        ].isna()
    )

    negative = (
        stock[
            "Source_Ending_Balance_2024"
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
            "MISSING_2024_ENDING_BALANCE",
            "NEGATIVE_2024_ENDING_BALANCE",
            "VALID_2024_ENDING_BALANCE",
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
    ] = stock.loc[
        valid,
        "Source_Ending_Balance_2024",
    ]

    return stock


# ============================================================
# LATEST VALID COST THROUGH 2024
# ============================================================

def build_cost(
    fact: pd.DataFrame,
):

    history = fact.copy()

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

    latest = (
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

    latest = latest.rename(
        columns={
            "Year":
                "Planning_Cost_Source_Year",

            "Average_cost":
                "Planning_Cost",

            "Source_File":
                "Planning_Cost_Source_File",
        }
    )

    medicines = (
        fact[
            [
                "Medicine_Key",
            ]
        ]
        .drop_duplicates()
    )

    result = medicines.merge(
        latest,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    result[
        "Planning_Cost_Available"
    ] = (
        result[
            "Planning_Cost"
        ]
        .notna()
        &
        result[
            "Planning_Cost"
        ]
        .gt(0)
    )

    result[
        "Planning_Cost_Fallback_Used"
    ] = (
        result[
            "Planning_Cost_Available"
        ]
        &
        result[
            "Planning_Cost_Source_Year"
        ]
        .lt(
            PLANNING_ORIGIN_YEAR
        )
    )

    result[
        "Planning_Cost_Status"
    ] = np.select(
        [
            ~result[
                "Planning_Cost_Available"
            ],

            result[
                "Planning_Cost_Source_Year"
            ]
            .eq(
                PLANNING_ORIGIN_YEAR
            ),

            result[
                "Planning_Cost_Source_Year"
            ]
            .lt(
                PLANNING_ORIGIN_YEAR
            ),
        ],
        [
            "UNAVAILABLE",
            "VALID_2024_COST",
            "HISTORICAL_VALID_COST_FALLBACK",
        ],
        default="UNKNOWN",
    )

    return result


# ============================================================
# BUILD RETROSPECTIVE DATASET
# ============================================================

def build_retrospective(
    supervised: pd.DataFrame,
    quantiles: pd.DataFrame,
    stock: pd.DataFrame,
    cost: pd.DataFrame,
):

    keep = [
        "Medicine_Key",
        "Brand_Name",
        "Generic_Name",
        "Unit",
        "Target_Year",
        "Feature_Cutoff_Year",
        "Max_History_Year_Used",
        "ABC_AsOf",
        "Behavior_Segment_AsOf",
        "Lag1_Issuance",
        "Target_Annual_Issuance",
    ]

    result = supervised[
        keep
    ].copy()

    result = result.rename(
        columns={
            "Target_Annual_Issuance":
                "Actual_2025_Annual_Issuance"
        }
    )

    result = result.merge(
        quantiles,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
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
    # GOVERNED POINT FORECAST
    # --------------------------------------------------------

    result[
        "Raw_Requirement_Point"
    ] = (
        result[
            "Lag1_Issuance"
        ]
        .astype(float)
    )

    # --------------------------------------------------------
    # PRESERVE RAW BACKTEST QUANTILES
    # --------------------------------------------------------

    result[
        "Backtest_P80"
    ] = (
        result[
            "P80"
        ]
    )

    result[
        "Backtest_P90"
    ] = (
        result[
            "P90"
        ]
    )

    # --------------------------------------------------------
    # PLANNING POLICIES
    # --------------------------------------------------------

    result[
        "Planning_Requirement_Point"
    ] = (
        np.round(
            result[
                "Raw_Requirement_Point"
            ]
        )
        .astype(
            np.int64
        )
    )

    result[
        "Planning_Requirement_P80_Floored_Ceil"
    ] = np.ceil(
        np.maximum(
            result[
                "Raw_Requirement_Point"
            ],
            result[
                "Backtest_P80"
            ],
        )
        -
        1e-9
    ).astype(
        np.int64
    )

    result[
        "Planning_Requirement_P90_Floored_Ceil"
    ] = np.ceil(
        np.maximum(
            result[
                "Raw_Requirement_Point"
            ],
            result[
                "Backtest_P90"
            ],
        )
        -
        1e-9
    ).astype(
        np.int64
    )

    result[
        "P80_Incumbent_Floor_Applied"
    ] = (
        result[
            "Backtest_P80"
        ]
        <
        result[
            "Raw_Requirement_Point"
        ]
    )

    result[
        "P90_Incumbent_Floor_Applied"
    ] = (
        result[
            "Backtest_P90"
        ]
        <
        result[
            "Raw_Requirement_Point"
        ]
    )

    # --------------------------------------------------------
    # ELIGIBILITY
    # --------------------------------------------------------

    stock_valid = (
        result[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    cost_valid = (
        result[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    abc_valid = (
        result[
            "ABC_AsOf"
        ]
        .isin(
            PRIORITY_CLASSES
        )
    )

    result[
        "Retrospective_Optimization_Eligible"
    ] = (
        stock_valid
        &
        cost_valid
        &
        abc_valid
    )

    result[
        "High_Priority_AB_Unresolved"
    ] = (
        result[
            "ABC_AsOf"
        ]
        .isin(
            [
                "A",
                "B",
            ]
        )
        &
        (
            ~stock_valid
            |
            ~cost_valid
        )
    )

    # --------------------------------------------------------
    # NET REQUIREMENTS
    # --------------------------------------------------------

    scenario_columns = {
        "POINT":
            "Planning_Requirement_Point",

        "P80_FLOORED_CEIL":
            "Planning_Requirement_P80_Floored_Ceil",

        "P90_FLOORED_CEIL":
            "Planning_Requirement_P90_Floored_Ceil",
    }

    for scenario, requirement_column in (
        scenario_columns.items()
    ):

        net_column = (
            f"Retrospective_Net_Requirement_{scenario}"
        )

        value_column = (
            f"Retrospective_B100_Value_{scenario}"
        )

        result[
            net_column
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

        valid_stock_rows = (
            stock_valid
        )

        requirement = (
            result.loc[
                valid_stock_rows,
                requirement_column,
            ]
            .astype(
                np.int64
            )
            .to_numpy()
        )

        stock_qty = (
            np.round(
                result.loc[
                    valid_stock_rows,
                    "Planning_Stock_Qty",
                ]
                .astype(float)
                .to_numpy()
            )
            .astype(
                np.int64
            )
        )

        result.loc[
            valid_stock_rows,
            net_column,
        ] = np.maximum(
            0,
            requirement
            -
            stock_qty,
        ).astype(
            np.int64
        )

        result[
            value_column
        ] = np.nan

        eligible = (
            result[
                "Retrospective_Optimization_Eligible"
            ]
        )

        result.loc[
            eligible,
            value_column,
        ] = (
            result.loc[
                eligible,
                net_column,
            ]
            .astype(float)
            *
            result.loc[
                eligible,
                "Planning_Cost",
            ]
            .astype(float)
        )

    # --------------------------------------------------------
    # RETROSPECTIVE GOVERNANCE
    # --------------------------------------------------------

    result[
        "Observed_2025_Used_For_Planning"
    ] = False

    result[
        "Observed_2025_Use"
    ] = (
        "RETROSPECTIVE_EVALUATION_ONLY"
    )

    result[
        "Planning_Origin"
    ] = (
        "END_OF_2024"
    )

    result[
        "Historical_ABC_Rewritten"
    ] = False

    result[
        "Last_Purchase_Price_Used"
    ] = False

    result[
        "Year_End_Implied_Cost_Used"
    ] = False

    result[
        "Retrospective_Version"
    ] = RETROSPECTIVE_VERSION

    return result


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_retrospective(
    dataframe: pd.DataFrame,
):

    if len(dataframe) != EXPECTED_MEDICINES:

        raise ValueError(
            "Retrospective row count failed."
        )

    if dataframe[
        "Medicine_Key"
    ].duplicated().any():

        raise ValueError(
            "Duplicate Medicine_Key."
        )

    if not dataframe[
        "Target_Year"
    ].eq(
        TARGET_YEAR
    ).all():

        raise ValueError(
            "Unexpected target year."
        )

    if not dataframe[
        "Feature_Cutoff_Year"
    ].eq(
        PLANNING_ORIGIN_YEAR
    ).all():

        raise ValueError(
            "Forecast cutoff leakage."
        )

    if not dataframe[
        "Max_History_Year_Used"
    ].eq(
        PLANNING_ORIGIN_YEAR
    ).all():

        raise ValueError(
            "Historical leakage detected."
        )

    if dataframe[
        "Observed_2025_Used_For_Planning"
    ].any():

        raise ValueError(
            "2025 outcome leakage flag TRUE."
        )

    # Point total must equal 2024
    # observed issuance.

    point_total = float(
        dataframe[
            "Planning_Requirement_Point"
        ].sum()
    )

    if abs(
        point_total
        -
        111479.0
    ) > 1e-6:

        raise ValueError(
            f"Unexpected 2025 point "
            f"forecast total: "
            f"{point_total:,.2f}"
        )

    actual_total = float(
        dataframe[
            "Actual_2025_Annual_Issuance"
        ].sum()
    )

    if abs(
        actual_total
        -
        128889.0
    ) > 1e-6:

        raise ValueError(
            f"Unexpected actual 2025 "
            f"issuance total: "
            f"{actual_total:,.2f}"
        )

    eligible = (
        dataframe[
            "Retrospective_Optimization_Eligible"
        ]
    )

    if not dataframe.loc[
        eligible,
        "ABC_AsOf",
    ].isin(
        PRIORITY_CLASSES
    ).all():

        raise ValueError(
            "Invalid ABC in retrospective "
            "optimizer population."
        )


# ============================================================
# OVERALL SUMMARY
# ============================================================

def build_summary(
    dataframe: pd.DataFrame,
):

    eligible = (
        dataframe[
            "Retrospective_Optimization_Eligible"
        ]
    )

    row = {
        "Target_Year":
            TARGET_YEAR,

        "Planning_Origin_Year":
            PLANNING_ORIGIN_YEAR,

        "Medicine_Count":
            len(
                dataframe
            ),

        "Actual_2025_Issuance_Total":
            float(
                dataframe[
                    "Actual_2025_Annual_Issuance"
                ].sum()
            ),

        "Point_Forecast_Total":
            float(
                dataframe[
                    "Planning_Requirement_Point"
                ].sum()
            ),

        "P80_Floored_Ceil_Total":
            float(
                dataframe[
                    "Planning_Requirement_P80_Floored_Ceil"
                ].sum()
            ),

        "P90_Floored_Ceil_Total":
            float(
                dataframe[
                    "Planning_Requirement_P90_Floored_Ceil"
                ].sum()
            ),

        "P80_Floor_Applied_Count":
            int(
                dataframe[
                    "P80_Incumbent_Floor_Applied"
                ].sum()
            ),

        "P90_Floor_Applied_Count":
            int(
                dataframe[
                    "P90_Incumbent_Floor_Applied"
                ].sum()
            ),

        "Valid_2024_Stock_Count":
            int(
                dataframe[
                    "Planning_Stock_Valid"
                ].sum()
            ),

        "Missing_2024_Stock_Count":
            int(
                dataframe[
                    "Planning_Stock_Status"
                ]
                .eq(
                    "MISSING_2024_ENDING_BALANCE"
                )
                .sum()
            ),

        "Negative_2024_Stock_Count":
            int(
                dataframe[
                    "Planning_Stock_Status"
                ]
                .eq(
                    "NEGATIVE_2024_ENDING_BALANCE"
                )
                .sum()
            ),

        "Valid_2024_Cost_Count":
            int(
                dataframe[
                    "Planning_Cost_Status"
                ]
                .eq(
                    "VALID_2024_COST"
                )
                .sum()
            ),

        "Historical_Cost_Fallback_Count":
            int(
                dataframe[
                    "Planning_Cost_Fallback_Used"
                ]
                .sum()
            ),

        "Cost_Unavailable_Count":
            int(
                (
                    ~dataframe[
                        "Planning_Cost_Available"
                    ]
                )
                .sum()
            ),

        "Optimizer_Eligible_Count":
            int(
                eligible.sum()
            ),

        "Optimizer_Excluded_Count":
            int(
                (
                    ~eligible
                )
                .sum()
            ),

        "High_Priority_AB_Unresolved_Count":
            int(
                dataframe[
                    "High_Priority_AB_Unresolved"
                ].sum()
            ),

        "Retrospective_Version":
            RETROSPECTIVE_VERSION,
    }

    return pd.DataFrame(
        [row]
    )


# ============================================================
# SCENARIO SUMMARY
# ============================================================

def build_scenario_summary(
    dataframe: pd.DataFrame,
):

    eligible = (
        dataframe[
            "Retrospective_Optimization_Eligible"
        ]
    )

    config = {
        "POINT": (
            "Planning_Requirement_Point",
            "Retrospective_Net_Requirement_POINT",
            "Retrospective_B100_Value_POINT",
        ),

        "P80_FLOORED_CEIL": (
            "Planning_Requirement_P80_Floored_Ceil",
            "Retrospective_Net_Requirement_P80_FLOORED_CEIL",
            "Retrospective_B100_Value_P80_FLOORED_CEIL",
        ),

        "P90_FLOORED_CEIL": (
            "Planning_Requirement_P90_Floored_Ceil",
            "Retrospective_Net_Requirement_P90_FLOORED_CEIL",
            "Retrospective_B100_Value_P90_FLOORED_CEIL",
        ),
    }

    rows = []

    for scenario, (
        requirement_column,
        net_column,
        value_column,
    ) in config.items():

        rows.append(
            {
                "Scenario":
                    scenario,

                "Eligible_Count":
                    int(
                        eligible.sum()
                    ),

                "Eligible_A_Count":
                    int(
                        (
                            eligible
                            &
                            dataframe[
                                "ABC_AsOf"
                            ].eq("A")
                        )
                        .sum()
                    ),

                "Eligible_B_Count":
                    int(
                        (
                            eligible
                            &
                            dataframe[
                                "ABC_AsOf"
                            ].eq("B")
                        )
                        .sum()
                    ),

                "Eligible_C_Count":
                    int(
                        (
                            eligible
                            &
                            dataframe[
                                "ABC_AsOf"
                            ].eq("C")
                        )
                        .sum()
                    ),

                "Gross_Requirement_Units":
                    int(
                        dataframe[
                            requirement_column
                        ].sum()
                    ),

                "Eligible_Net_Requirement_Units":
                    int(
                        dataframe.loc[
                            eligible,
                            net_column,
                        ]
                        .astype(
                            np.int64
                        )
                        .sum()
                    ),

                "Positive_Net_Eligible_Count":
                    int(
                        dataframe.loc[
                            eligible,
                            net_column,
                        ]
                        .astype(
                            np.int64
                        )
                        .gt(0)
                        .sum()
                    ),

                "Retrospective_B100":
                    float(
                        dataframe.loc[
                            eligible,
                            value_column,
                        ]
                        .sum()
                    ),

                "Actual_2025_Issuance_Total":
                    float(
                        dataframe[
                            "Actual_2025_Annual_Issuance"
                        ].sum()
                    ),

                "Actual_2025_Used_To_Create_Requirement":
                    False,

                "Retrospective_Version":
                    RETROSPECTIVE_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BY ABC READINESS
# ============================================================

def build_by_abc(
    dataframe: pd.DataFrame,
):

    rows = []

    for abc_class, group in (
        dataframe.groupby(
            "ABC_AsOf",
            dropna=False,
            sort=True,
        )
    ):

        name = (
            str(abc_class)
            if pd.notna(
                abc_class
            )
            else "MISSING"
        )

        rows.append(
            {
                "ABC_AsOf":
                    name,

                "Medicine_Count":
                    len(
                        group
                    ),

                "Optimizer_Eligible_Count":
                    int(
                        group[
                            "Retrospective_Optimization_Eligible"
                        ].sum()
                    ),

                "Stock_Review_Count":
                    int(
                        (
                            ~group[
                                "Planning_Stock_Valid"
                            ]
                        )
                        .sum()
                    ),

                "Cost_Review_Count":
                    int(
                        (
                            ~group[
                                "Planning_Cost_Available"
                            ]
                        )
                        .sum()
                    ),

                "High_Priority_Unresolved_Count":
                    int(
                        group[
                            "High_Priority_AB_Unresolved"
                        ].sum()
                    ),

                "Actual_2025_Issuance":
                    float(
                        group[
                            "Actual_2025_Annual_Issuance"
                        ].sum()
                    ),

                "Point_Requirement":
                    float(
                        group[
                            "Planning_Requirement_Point"
                        ].sum()
                    ),

                "P80_Requirement":
                    float(
                        group[
                            "Planning_Requirement_P80_Floored_Ceil"
                        ].sum()
                    ),

                "P90_Requirement":
                    float(
                        group[
                            "Planning_Requirement_P90_Floored_Ceil"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance():

    return pd.DataFrame(
        [
            {
                "Retrospective_Version":
                    RETROSPECTIVE_VERSION,

                "Planning_Origin":
                    "END_OF_2024",

                "Target_Year":
                    2025,

                "Forecast_Features_Through":
                    2024,

                "Baseline_Stock":
                    (
                        "2024 annual Ending_Balance "
                        "when observed and nonnegative"
                    ),

                "Planning_Cost":
                    (
                        "Latest valid annual "
                        "Average_cost through 2024"
                    ),

                "Point_Forecast":
                    "2024 annual issuance",

                "P80_P90_Source":
                    (
                        "Chronological BACKTEST_2025 "
                        "quantile forecasts"
                    ),

                "Actual_2025_Issuance_Used_For_Planning":
                    False,

                "Actual_2025_Issuance_Used_For_Evaluation":
                    True,

                "Random_Split":
                    False,

                "Actual_Stockout_Measured":
                    False,

                "Actual_Waste_Measured":
                    False,

                "Simulation_Outcome_Names":
                    (
                        "Simulated annual demand "
                        "shortfall / simulated "
                        "residual inventory"
                    ),

                "Optimizer_Run":
                    False,
            }
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    inputs,
    summary,
    scenarios,
    by_abc,
    governance,
):

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "inputs_df",
            inputs,
        )

        connection.register(
            "summary_df",
            summary,
        )

        connection.register(
            "scenarios_df",
            scenarios,
        )

        connection.register(
            "abc_df",
            by_abc,
        )

        connection.register(
            "gov_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_inputs AS
            SELECT * FROM inputs_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_readiness_summary AS
            SELECT * FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_scenario_summary AS
            SELECT * FROM scenarios_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_readiness_by_abc AS
            SELECT * FROM abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_governance AS
            SELECT * FROM gov_df
            """
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 116)
    print("CDH INVENTORY DSS")
    print("2025 RETROSPECTIVE PLANNING-ORIGIN RECONSTRUCTION")
    print("=" * 116)
    print()

    (
        supervised,
        quantiles,
        fact,
    ) = load_data()

    validate_sources(
        supervised,
        quantiles,
        fact,
    )

    print(
        "Temporal source validation: PASS"
    )

    print(
        "Planning origin: end of 2024"
    )

    print(
        "Target year: 2025"
    )

    print(
        "Observed 2025 issuance used "
        "for planning: NO"
    )

    stock = build_stock(
        fact
    )

    cost = build_cost(
        fact
    )

    retrospective = (
        build_retrospective(
            supervised,
            quantiles,
            stock,
            cost,
        )
    )

    validate_retrospective(
        retrospective
    )

    print(
        "Retrospective structural "
        "validation: PASS"
    )

    summary = build_summary(
        retrospective
    )

    scenarios = (
        build_scenario_summary(
            retrospective
        )
    )

    by_abc = build_by_abc(
        retrospective
    )

    governance = (
        build_governance()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    retrospective.to_parquet(
        OUTPUT_DIR
        / "retrospective_2025_inputs.parquet",
        index=False,
        engine="pyarrow",
    )

    retrospective.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_inputs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_readiness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scenarios.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_readiness_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        retrospective,
        summary,
        scenarios,
        by_abc,
        governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 116)
    print("RETROSPECTIVE READINESS SUMMARY")
    print("-" * 116)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 116)
    print("RETROSPECTIVE REQUIREMENT SCENARIOS")
    print("-" * 116)

    print(
        scenarios.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 116)
    print("RETROSPECTIVE READINESS BY AS-OF ABC")
    print("-" * 116)

    print(
        by_abc.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 116)
    print("RETROSPECTIVE GOVERNANCE")
    print("-" * 116)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 116)
    print("IMPORTANT")
    print("-" * 116)

    print(
        "All forecast and planning "
        "inputs were available by the "
        "end of 2024."
    )

    print(
        "Actual 2025 issuance is retained "
        "only as an evaluation outcome."
    )

    print(
        "No actual 2025 delivery, ending "
        "balance, cost, or contemporaneous "
        "ABC value is used to construct "
        "the retrospective procurement plan."
    )

    print(
        "The next step may simulate annual "
        "availability as 2024 ending stock "
        "+ modeled procurement allocation."
    )

    print(
        "Any resulting shortage measure "
        "will be called simulated annual "
        "demand shortfall, not an actual "
        "historical stockout."
    )

    print(
        "Any remaining quantity will be "
        "called simulated residual inventory, "
        "not actual waste."
    )

    print(
        "No retrospective optimizer has "
        "been executed yet."
    )

    print()
    print("=" * 116)
    print(
        "2025 RETROSPECTIVE INPUT "
        "RECONSTRUCTION COMPLETE"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()