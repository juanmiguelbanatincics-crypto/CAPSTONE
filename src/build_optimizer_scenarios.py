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
    / "optimizer_scenarios_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

SCENARIO_VERSION = "OPTIMIZER_SCENARIOS_2026_V1"

EXPECTED_MEDICINES = 2733

PRIORITY_CLASSES = [
    "A",
    "B",
    "C",
]

BUDGET_FACTORS = [
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

ABC_PRIORITY_ORDER = {
    "A": 1,
    "B": 2,
    "C": 3,
    "UNCLASSIFIED_COST": 4,
}


# ============================================================
# LOAD PLANNING DATA
# ============================================================

def load_planning() -> pd.DataFrame:

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
# SOURCE VALIDATION
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
            "Medicine_Key uniqueness failed."
        )


# ============================================================
# BUILD GOVERNED REQUIREMENTS
# ============================================================

def build_optimizer_inputs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    # --------------------------------------------------------
    # PRESERVE RAW FORECAST OUTPUTS
    # --------------------------------------------------------

    result[
        "Raw_Requirement_Point"
    ] = (
        result[
            "Requirement_Point"
        ]
    )

    result[
        "Raw_Requirement_P80"
    ] = (
        result[
            "Requirement_P80"
        ]
    )

    result[
        "Raw_Requirement_P90"
    ] = (
        result[
            "Requirement_P90"
        ]
    )

    # --------------------------------------------------------
    # GOVERNED DECISION-POLICY REQUIREMENTS
    # --------------------------------------------------------

    result[
        "Planning_Requirement_Point"
    ] = (
        result[
            "Raw_Requirement_Point"
        ]
    )

    result[
        "Planning_Requirement_P80_Floored"
    ] = np.maximum(
        result[
            "Raw_Requirement_Point"
        ],
        result[
            "Raw_Requirement_P80"
        ],
    )

    result[
        "Planning_Requirement_P90_Floored"
    ] = np.maximum(
        result[
            "Raw_Requirement_Point"
        ],
        result[
            "Raw_Requirement_P90"
        ],
    )

    result[
        "P80_Incumbent_Floor_Applied"
    ] = (
        result[
            "Raw_Requirement_P80"
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
            "Raw_Requirement_P90"
        ]
        <
        result[
            "Raw_Requirement_Point"
        ]
    )

    # --------------------------------------------------------
    # INPUT ELIGIBILITY
    # --------------------------------------------------------

    core_ready = (
        result[
            "Optimizer_Core_Input_Ready"
        ]
        .fillna(False)
        .astype(bool)
    )

    abc_classifiable = (
        result[
            "ABC_AsOf"
        ]
        .isin(
            PRIORITY_CLASSES
        )
    )

    result[
        "Optimization_Priority_Eligible"
    ] = (
        core_ready
        &
        abc_classifiable
    )

    result[
        "Ready_But_ABC_Unclassified"
    ] = (
        core_ready
        &
        result[
            "ABC_AsOf"
        ]
        .eq(
            "UNCLASSIFIED_COST"
        )
    )

    result[
        "High_Priority_AB_Unresolved"
    ] = (
        ~core_ready
        &
        result[
            "ABC_AsOf"
        ]
        .isin(
            [
                "A",
                "B",
            ]
        )
    )

    # --------------------------------------------------------
    # EXCLUSION REASON
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

    result[
        "Optimization_Eligibility_Status"
    ] = np.select(
        [
            (
                core_ready
                &
                abc_classifiable
            ),

            (
                core_ready
                &
                result[
                    "ABC_AsOf"
                ]
                .eq(
                    "UNCLASSIFIED_COST"
                )
            ),

            (
                ~stock_valid
                &
                cost_valid
            ),

            (
                stock_valid
                &
                ~cost_valid
            ),

            (
                ~stock_valid
                &
                ~cost_valid
            ),
        ],
        [
            "ELIGIBLE_ABC_PRIORITY",
            "READY_BUT_ABC_UNCLASSIFIED",
            "EXCLUDED_STOCK_REVIEW",
            "EXCLUDED_COST_REVIEW",
            "EXCLUDED_STOCK_AND_COST_REVIEW",
        ],
        default="UNKNOWN",
    )

    # --------------------------------------------------------
    # ABC PRIORITY
    # --------------------------------------------------------

    result[
        "ABC_Priority_Order"
    ] = (
        result[
            "ABC_AsOf"
        ]
        .map(
            ABC_PRIORITY_ORDER
        )
        .fillna(99)
        .astype(int)
    )

    # --------------------------------------------------------
    # NET REQUIREMENTS
    # --------------------------------------------------------

    requirement_map = {
        "POINT":
            "Planning_Requirement_Point",

        "P80_FLOORED":
            "Planning_Requirement_P80_Floored",

        "P90_FLOORED":
            "Planning_Requirement_P90_Floored",
    }

    for scenario, requirement_column in (
        requirement_map.items()
    ):

        net_column = (
            f"Optimization_Net_"
            f"Requirement_{scenario}"
        )

        value_column = (
            f"Optimization_Full_"
            f"Funding_Value_{scenario}"
        )

        result[
            net_column
        ] = np.nan

        # Net requirement can only be
        # calculated when stock is valid.
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

        result[
            value_column
        ] = np.nan

        eligible = (
            result[
                "Optimization_Priority_Eligible"
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
            *
            result.loc[
                eligible,
                "Planning_Cost",
            ]
        )

    # --------------------------------------------------------
    # DECISION-POLICY LABELS
    # --------------------------------------------------------

    result[
        "Point_Requirement_Policy"
    ] = (
        "GOVERNED_NAIVE_INCUMBENT"
    )

    result[
        "P80_Requirement_Policy"
    ] = (
        "MAX_INCUMBENT_AND_RAW_P80"
    )

    result[
        "P90_Requirement_Policy"
    ] = (
        "MAX_INCUMBENT_AND_RAW_P90"
    )

    result[
        "Raw_P80_Remains_Statistical_Quantile"
    ] = True

    result[
        "Raw_P90_Remains_Statistical_Quantile"
    ] = True

    result[
        "Floored_P80_Is_Statistical_Quantile"
    ] = False

    result[
        "Floored_P90_Is_Statistical_Quantile"
    ] = False

    result[
        "Scenario_Version"
    ] = SCENARIO_VERSION

    return result


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_optimizer_inputs(
    dataframe: pd.DataFrame,
) -> None:

    if len(
        dataframe
    ) != EXPECTED_MEDICINES:

        raise ValueError(
            "Optimizer-input row count failed."
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate Medicine_Key."
        )

    if (
        dataframe[
            "Planning_Requirement_P80_Floored"
        ]
        .lt(
            dataframe[
                "Planning_Requirement_Point"
            ]
        )
        .any()
    ):

        raise ValueError(
            "Floored P80 below point forecast."
        )

    if (
        dataframe[
            "Planning_Requirement_P90_Floored"
        ]
        .lt(
            dataframe[
                "Planning_Requirement_Point"
            ]
        )
        .any()
    ):

        raise ValueError(
            "Floored P90 below point forecast."
        )

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
    )

    if (
        ~dataframe.loc[
            eligible,
            "ABC_AsOf",
        ]
        .isin(
            PRIORITY_CLASSES
        )
    ).any():

        raise ValueError(
            "Invalid ABC class entered "
            "priority-eligible population."
        )

    if (
        ~dataframe.loc[
            eligible,
            "Planning_Stock_Valid",
        ]
        .fillna(False)
        .astype(bool)
    ).any():

        raise ValueError(
            "Eligible row lacks valid stock."
        )

    if (
        ~dataframe.loc[
            eligible,
            "Planning_Cost_Available",
        ]
        .fillna(False)
        .astype(bool)
    ).any():

        raise ValueError(
            "Eligible row lacks planning cost."
        )

    for scenario in [
        "POINT",
        "P80_FLOORED",
        "P90_FLOORED",
    ]:

        net_column = (
            f"Optimization_Net_"
            f"Requirement_{scenario}"
        )

        value_column = (
            f"Optimization_Full_"
            f"Funding_Value_{scenario}"
        )

        if (
            dataframe.loc[
                eligible,
                net_column,
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                f"Missing eligible "
                f"{net_column}."
            )

        if (
            dataframe.loc[
                eligible,
                net_column,
            ]
            .lt(0)
            .any()
        ):

            raise ValueError(
                f"Negative {net_column}."
            )

        if (
            dataframe.loc[
                eligible,
                value_column,
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                f"Missing eligible "
                f"{value_column}."
            )


# ============================================================
# SCENARIO SUMMARY
# ============================================================

def build_scenario_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
    )

    unresolved_ab = (
        dataframe[
            "High_Priority_AB_Unresolved"
        ]
    )

    ready_unclassified = (
        dataframe[
            "Ready_But_ABC_Unclassified"
        ]
    )

    rows = []

    scenarios = {
        "POINT": {
            "raw":
                "Raw_Requirement_Point",

            "planning":
                "Planning_Requirement_Point",

            "net":
                "Optimization_Net_Requirement_POINT",

            "value":
                "Optimization_Full_Funding_Value_POINT",
        },

        "P80_FLOORED": {
            "raw":
                "Raw_Requirement_P80",

            "planning":
                "Planning_Requirement_P80_Floored",

            "net":
                "Optimization_Net_Requirement_P80_FLOORED",

            "value":
                "Optimization_Full_Funding_Value_P80_FLOORED",
        },

        "P90_FLOORED": {
            "raw":
                "Raw_Requirement_P90",

            "planning":
                "Planning_Requirement_P90_Floored",

            "net":
                "Optimization_Net_Requirement_P90_FLOORED",

            "value":
                "Optimization_Full_Funding_Value_P90_FLOORED",
        },
    }

    for scenario, columns in (
        scenarios.items()
    ):

        rows.append(
            {
                "Scenario":
                    scenario,

                "Total_Medicines":
                    len(
                        dataframe
                    ),

                "Priority_Eligible_Count":
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
                            ]
                            .eq("A")
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
                            ]
                            .eq("B")
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
                            ]
                            .eq("C")
                        )
                        .sum()
                    ),

                "High_Priority_AB_Unresolved_Count":
                    int(
                        unresolved_ab.sum()
                    ),

                "Ready_But_ABC_Unclassified_Count":
                    int(
                        ready_unclassified.sum()
                    ),

                "Raw_Requirement_Total":
                    float(
                        dataframe[
                            columns[
                                "raw"
                            ]
                        ]
                        .sum()
                    ),

                "Governed_Planning_Requirement_Total":
                    float(
                        dataframe[
                            columns[
                                "planning"
                            ]
                        ]
                        .sum()
                    ),

                "Eligible_Net_Requirement_Total":
                    float(
                        dataframe.loc[
                            eligible,
                            columns[
                                "net"
                            ],
                        ]
                        .sum()
                    ),

                "Eligible_B100":
                    float(
                        dataframe.loc[
                            eligible,
                            columns[
                                "value"
                            ],
                        ]
                        .sum()
                    ),

                "Full_AB_Population_Represented":
                    bool(
                        unresolved_ab.sum()
                        == 0
                    ),

                "Scenario_Scope":
                    (
                        "ELIGIBLE_ABC_RECORDS_ONLY"
                    ),

                "Scenario_Version":
                    SCENARIO_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BY ABC SUMMARY
# ============================================================

def build_by_abc_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for abc_class in [
        "A",
        "B",
        "C",
        "UNCLASSIFIED_COST",
    ]:

        group = (
            dataframe.loc[
                dataframe[
                    "ABC_AsOf"
                ]
                .eq(
                    abc_class
                )
            ]
        )

        eligible = (
            group[
                "Optimization_Priority_Eligible"
            ]
        )

        rows.append(
            {
                "ABC_AsOf":
                    abc_class,

                "Medicine_Count":
                    len(group),

                "Priority_Eligible_Count":
                    int(
                        eligible.sum()
                    ),

                "Excluded_Count":
                    int(
                        (
                            ~eligible
                        )
                        .sum()
                    ),

                "Unresolved_Stock_Count":
                    int(
                        (
                            ~group[
                                "Planning_Stock_Valid"
                            ]
                            .fillna(False)
                            .astype(bool)
                        )
                        .sum()
                    ),

                "Unresolved_Cost_Count":
                    int(
                        (
                            ~group[
                                "Planning_Cost_Available"
                            ]
                            .fillna(False)
                            .astype(bool)
                        )
                        .sum()
                    ),

                "P80_Floor_Applied_Count":
                    int(
                        group[
                            "P80_Incumbent_Floor_Applied"
                        ]
                        .sum()
                    ),

                "P90_Floor_Applied_Count":
                    int(
                        group[
                            "P90_Incumbent_Floor_Applied"
                        ]
                        .sum()
                    ),

                "Planning_Point_Total":
                    float(
                        group[
                            "Planning_Requirement_Point"
                        ]
                        .sum()
                    ),

                "Planning_P80_Floored_Total":
                    float(
                        group[
                            "Planning_Requirement_P80_Floored"
                        ]
                        .sum()
                    ),

                "Planning_P90_Floored_Total":
                    float(
                        group[
                            "Planning_Requirement_P90_Floored"
                        ]
                        .sum()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BUDGET SCENARIOS
# ============================================================

def build_budget_scenarios(
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in (
        scenario_summary.iterrows()
    ):

        eligible_b100 = float(
            row[
                "Eligible_B100"
            ]
        )

        for factor in BUDGET_FACTORS:

            rows.append(
                {
                    "Requirement_Scenario":
                        row[
                            "Scenario"
                        ],

                    "Budget_Factor":
                        factor,

                    "Budget_Percent":
                        factor
                        * 100,

                    "Eligible_B100":
                        eligible_b100,

                    "Scenario_Budget":
                        eligible_b100
                        * factor,

                    "Budget_Source":
                        (
                            "MODELED_PERCENT_OF_"
                            "ELIGIBLE_FULL_"
                            "FUNDING_REQUIREMENT"
                        ),

                    "Actual_CDH_Budget":
                        False,

                    "Scenario_Scope":
                        (
                            "ELIGIBLE_ABC_RECORDS_ONLY"
                        ),

                    "Scenario_Version":
                        SCENARIO_VERSION,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# QUANTITY GRANULARITY DIAGNOSTIC
# ============================================================

def non_integer_count(
    series: pd.Series,
    tolerance: float = 1e-9,
) -> int:

    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    return int(
        (
            np.abs(
                values
                -
                np.round(
                    values
                )
            )
            >
            tolerance
        )
        .sum()
    )


def build_quantity_diagnostic(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    valid_stock = (
        dataframe.loc[
            dataframe[
                "Planning_Stock_Valid"
            ]
            .fillna(False)
            .astype(bool),
            "Planning_Stock_Qty",
        ]
    )

    return pd.DataFrame(
        [
            {
                "Valid_Stock_Rows":
                    len(
                        valid_stock
                    ),

                "NonInteger_Valid_Stock_Count":
                    non_integer_count(
                        valid_stock
                    ),

                "NonInteger_Point_Requirement_Count":
                    non_integer_count(
                        dataframe[
                            "Planning_Requirement_Point"
                        ]
                    ),

                "NonInteger_P80_Floored_Count":
                    non_integer_count(
                        dataframe[
                            "Planning_Requirement_P80_Floored"
                        ]
                    ),

                "NonInteger_P90_Floored_Count":
                    non_integer_count(
                        dataframe[
                            "Planning_Requirement_P90_Floored"
                        ]
                    ),

                "Decision_Variable_Integrality":
                    "NOT_YET_DECIDED",

                "Reason":
                    (
                        "Quantity granularity is "
                        "diagnosed before deciding "
                        "whether procurement x_i "
                        "should be continuous or integer."
                    ),

                "Scenario_Version":
                    SCENARIO_VERSION,
            }
        ]
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    unresolved_ab = int(
        dataframe[
            "High_Priority_AB_Unresolved"
        ]
        .sum()
    )

    return pd.DataFrame(
        [
            {
                "Scenario_Version":
                    SCENARIO_VERSION,

                "Target_Year":
                    2026,

                "Priority_Classes":
                    "A>B>C",

                "Optimization_Scope":
                    "ELIGIBLE_ABC_RECORDS_ONLY",

                "Priority_Eligible_Definition":
                    (
                        "Valid planning stock + "
                        "available planning cost + "
                        "ABC class A/B/C."
                    ),

                "Unclassified_Cost_Reassigned_To_C":
                    False,

                "P80_Planning_Rule":
                    (
                        "max(NAIVE incumbent, raw P80)"
                    ),

                "P90_Planning_Rule":
                    (
                        "max(NAIVE incumbent, raw P90)"
                    ),

                "Raw_Quantiles_Modified":
                    False,

                "Floored_Requirements_Are_Quantiles":
                    False,

                "High_Priority_AB_Unresolved_Count":
                    unresolved_ab,

                "Full_High_Priority_Optimization_Claim_Allowed":
                    bool(
                        unresolved_ab
                        == 0
                    ),

                "Budget_Scenarios_Are_Actual_CDH_Budget":
                    False,

                "Optimizer_Executed":
                    False,
            }
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    inputs: pd.DataFrame,
    summary: pd.DataFrame,
    by_abc: pd.DataFrame,
    budgets: pd.DataFrame,
    diagnostic: pd.DataFrame,
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
            "summary_df",
            summary,
        )

        connection.register(
            "by_abc_df",
            by_abc,
        )

        connection.register(
            "budgets_df",
            budgets,
        )

        connection.register(
            "diagnostic_df",
            diagnostic,
        )

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_optimizer_inputs AS
            SELECT *
            FROM inputs_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_optimizer_scenario_summary AS
            SELECT *
            FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_optimizer_by_abc AS
            SELECT *
            FROM by_abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_budget_scenarios AS
            SELECT *
            FROM budgets_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_quantity_diagnostic AS
            SELECT *
            FROM diagnostic_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_optimizer_governance AS
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

    print("=" * 112)
    print("CDH INVENTORY DSS")
    print("2026 GOVERNED OPTIMIZATION SCENARIO INPUTS")
    print("=" * 112)
    print()

    planning = (
        load_planning()
    )

    validate_source(
        planning
    )

    print(
        "Planning source validation: PASS"
    )

    inputs = (
        build_optimizer_inputs(
            planning
        )
    )

    validate_optimizer_inputs(
        inputs
    )

    print(
        "Optimizer scenario structural "
        "validation: PASS"
    )

    summary = (
        build_scenario_summary(
            inputs
        )
    )

    by_abc = (
        build_by_abc_summary(
            inputs
        )
    )

    budgets = (
        build_budget_scenarios(
            summary
        )
    )

    diagnostic = (
        build_quantity_diagnostic(
            inputs
        )
    )

    governance = (
        build_governance(
            inputs
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    inputs.to_parquet(
        OUTPUT_DIR
        / "planning_2026_optimizer_inputs.parquet",
        index=False,
        engine="pyarrow",
    )

    inputs.to_csv(
        OUTPUT_DIR
        / "planning_2026_optimizer_inputs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "planning_2026_optimizer_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "planning_2026_optimizer_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    budgets.to_csv(
        OUTPUT_DIR
        / "planning_2026_budget_scenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostic.to_csv(
        OUTPUT_DIR
        / "planning_2026_quantity_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "planning_2026_optimizer_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        inputs=inputs,
        summary=summary,
        by_abc=by_abc,
        budgets=budgets,
        diagnostic=diagnostic,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 112)
    print("OPTIMIZATION SCENARIO SUMMARY")
    print("-" * 112)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 112)
    print("OPTIMIZER POPULATION BY ABC")
    print("-" * 112)

    print(
        by_abc.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 112)
    print("MODELED BUDGET SCENARIOS")
    print("-" * 112)

    print(
        budgets.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 112)
    print("QUANTITY GRANULARITY DIAGNOSTIC")
    print("-" * 112)

    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print("-" * 112)
    print("OPTIMIZER GOVERNANCE")
    print("-" * 112)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 112)
    print("IMPORTANT")
    print("-" * 112)

    print(
        "Raw P80/P90 forecasts remain "
        "unchanged statistical outputs."
    )

    print(
        "Floored P80/P90 are separate "
        "decision-policy requirements and "
        "must not be described as raw "
        "statistical quantiles."
    )

    print(
        "UNCLASSIFIED_COST medicines are "
        "not silently reassigned to C."
    )

    print(
        "The optimization population is "
        "limited to source-complete A/B/C "
        "records."
    )

    print(
        "Because unresolved A/B records "
        "remain, any subsequent optimization "
        "must be labeled an eligible-record "
        "scenario rather than a complete "
        "hospital procurement solution."
    )

    print(
        "The 60%-100% budgets are modeled "
        "scenarios, not actual CDH budgets."
    )

    print(
        "No optimizer has been executed."
    )

    print()
    print("=" * 112)
    print(
        "GOVERNED OPTIMIZATION SCENARIOS COMPLETE"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()