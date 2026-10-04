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
    / "milp_inputs_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

MILP_INPUT_VERSION = "MILP_INPUTS_2026_V1"

EXPECTED_MEDICINES = 2733

PRIORITY_CLASSES = [
    "A",
    "B",
    "C",
]

SCENARIOS = {
    "POINT": {
        "source_requirement":
            "Planning_Requirement_Point",
        "rounding":
            "EXACT_WHOLE_UNIT",
    },

    "P80_FLOORED_CEIL": {
        "source_requirement":
            "Planning_Requirement_P80_Floored",
        "rounding":
            "CEILING_TO_WHOLE_UNIT",
    },

    "P90_FLOORED_CEIL": {
        "source_requirement":
            "Planning_Requirement_P90_Floored",
        "rounding":
            "CEILING_TO_WHOLE_UNIT",
    },
}

BUDGET_FACTORS = [
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

INTEGER_TOLERANCE = 1e-9


# ============================================================
# LOAD DATA
# ============================================================

def load_optimizer_inputs() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            """
            SELECT *
            FROM planning_2026_optimizer_inputs
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
            f"rows; found {len(dataframe):,}"
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

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    if not (
        dataframe.loc[
            eligible,
            "ABC_AsOf",
        ]
        .isin(
            PRIORITY_CLASSES
        )
        .all()
    ):

        raise ValueError(
            "Non-A/B/C item found in "
            "priority-eligible population."
        )

    if (
        dataframe.loc[
            eligible,
            "Planning_Cost",
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Eligible row has missing cost."
        )

    if (
        dataframe.loc[
            eligible,
            "Planning_Cost",
        ]
        .le(0)
        .any()
    ):

        raise ValueError(
            "Eligible row has "
            "non-positive cost."
        )

    if (
        dataframe.loc[
            eligible,
            "Planning_Stock_Qty",
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Eligible row has missing stock."
        )


# ============================================================
# WHOLE-UNIT HELPERS
# ============================================================

def is_integer_like(
    values: np.ndarray,
) -> np.ndarray:

    return (
        np.abs(
            values
            -
            np.round(
                values
            )
        )
        <=
        INTEGER_TOLERANCE
    )


def ceiling_whole_units(
    values: np.ndarray,
) -> np.ndarray:

    # Subtract a tiny tolerance before
    # ceiling to avoid converting a value
    # such as 100.00000000000001 to 101.

    return np.ceil(
        values
        -
        INTEGER_TOLERANCE
    ).astype(
        np.int64
    )


# ============================================================
# BUILD MILP INPUTS
# ============================================================

def build_milp_inputs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    eligible = (
        result[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # STOCK GRANULARITY
    # --------------------------------------------------------

    eligible_stock = (
        result.loc[
            eligible,
            "Planning_Stock_Qty",
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not (
        is_integer_like(
            eligible_stock
        )
        .all()
    ):

        raise ValueError(
            "Eligible stock contains "
            "fractional quantities. "
            "MILP whole-unit policy cannot "
            "be applied automatically."
        )

    result[
        "MILP_Stock_Qty"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result.loc[
        eligible,
        "MILP_Stock_Qty",
    ] = (
        np.round(
            eligible_stock
        )
        .astype(
            np.int64
        )
    )

    # --------------------------------------------------------
    # REQUIREMENT SCENARIOS
    # --------------------------------------------------------

    for scenario, config in (
        SCENARIOS.items()
    ):

        source_column = (
            config[
                "source_requirement"
            ]
        )

        requirement_column = (
            f"MILP_Requirement_{scenario}"
        )

        net_column = (
            f"MILP_Net_Requirement_{scenario}"
        )

        value_column = (
            f"MILP_Full_Funding_Value_{scenario}"
        )

        increment_column = (
            f"MILP_Rounding_Increment_{scenario}"
        )

        source_values = (
            pd.to_numeric(
                result[
                    source_column
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
        )

        if not (
            np.isfinite(
                source_values
            )
            .all()
        ):

            raise ValueError(
                f"{source_column} contains "
                "missing/non-finite values."
            )

        if (
            source_values
            < 0
        ).any():

            raise ValueError(
                f"{source_column} contains "
                "negative requirements."
            )

        if (
            config[
                "rounding"
            ]
            ==
            "EXACT_WHOLE_UNIT"
        ):

            if not (
                is_integer_like(
                    source_values
                )
                .all()
            ):

                raise ValueError(
                    "Point planning requirement "
                    "is not whole-unit."
                )

            milp_requirement = (
                np.round(
                    source_values
                )
                .astype(
                    np.int64
                )
            )

        else:

            milp_requirement = (
                ceiling_whole_units(
                    source_values
                )
            )

        result[
            requirement_column
        ] = milp_requirement

        result[
            increment_column
        ] = (
            milp_requirement
            -
            source_values
        )

        # -----------------------------------------------
        # NET REQUIREMENT
        # -----------------------------------------------

        result[
            net_column
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

        eligible_requirement = (
            result.loc[
                eligible,
                requirement_column,
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        eligible_stock_integer = (
            result.loc[
                eligible,
                "MILP_Stock_Qty",
            ]
            .astype(
                np.int64
            )
            .to_numpy()
        )

        net_requirement = np.maximum(
            0,
            (
                eligible_requirement
                -
                eligible_stock_integer
            ),
        ).astype(
            np.int64
        )

        result.loc[
            eligible,
            net_column,
        ] = net_requirement

        # -----------------------------------------------
        # FULL-FUNDING VALUE
        # -----------------------------------------------

        result[
            value_column
        ] = np.nan

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

        # -----------------------------------------------
        # ACTIVE DECISION ROW
        # -----------------------------------------------

        result[
            f"MILP_Positive_Net_{scenario}"
        ] = False

        result.loc[
            eligible,
            f"MILP_Positive_Net_{scenario}",
        ] = (
            result.loc[
                eligible,
                net_column,
            ]
            .astype(
                np.int64
            )
            .gt(0)
        )

    # --------------------------------------------------------
    # DECISION VARIABLE GOVERNANCE
    # --------------------------------------------------------

    result[
        "MILP_Decision_Variable_Domain"
    ] = np.where(
        eligible,
        "NONNEGATIVE_INTEGER",
        "NOT_IN_OPTIMIZATION",
    )

    result[
        "MILP_Decision_Unit"
    ] = (
        "INVENTORY_REPORT_UNIT"
    )

    result[
        "Supplier_Pack_Size_Constraint_Available"
    ] = False

    result[
        "Clinical_Priority_Used"
    ] = False

    result[
        "MILP_Input_Version"
    ] = MILP_INPUT_VERSION

    return result


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_milp_inputs(
    dataframe: pd.DataFrame,
) -> None:

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    expected_eligible = int(
        eligible.sum()
    )

    if expected_eligible != 2365:

        raise ValueError(
            f"Expected 2,365 eligible "
            f"A/B/C rows; found "
            f"{expected_eligible:,}"
        )

    if (
        dataframe.loc[
            eligible,
            "MILP_Stock_Qty",
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "MILP stock missing on "
            "eligible record."
        )

    for scenario in SCENARIOS:

        requirement_column = (
            f"MILP_Requirement_{scenario}"
        )

        net_column = (
            f"MILP_Net_Requirement_{scenario}"
        )

        value_column = (
            f"MILP_Full_Funding_Value_{scenario}"
        )

        if (
            dataframe[
                requirement_column
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                f"Missing {requirement_column}."
            )

        if (
            dataframe[
                requirement_column
            ]
            .lt(0)
            .any()
        ):

            raise ValueError(
                f"Negative {requirement_column}."
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
            .astype(float)
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

    # Whole-unit scenario ordering.

    if (
        dataframe[
            "MILP_Requirement_P80_FLOORED_CEIL"
        ]
        .lt(
            dataframe[
                "MILP_Requirement_POINT"
            ]
        )
        .any()
    ):

        raise ValueError(
            "MILP P80 requirement "
            "below point requirement."
        )

    if (
        dataframe[
            "MILP_Requirement_P90_FLOORED_CEIL"
        ]
        .lt(
            dataframe[
                "MILP_Requirement_POINT"
            ]
        )
        .any()
    ):

        raise ValueError(
            "MILP P90 requirement "
            "below point requirement."
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
        .fillna(False)
        .astype(bool)
    )

    rows = []

    for scenario in SCENARIOS:

        requirement_column = (
            f"MILP_Requirement_{scenario}"
        )

        net_column = (
            f"MILP_Net_Requirement_{scenario}"
        )

        value_column = (
            f"MILP_Full_Funding_Value_{scenario}"
        )

        active_column = (
            f"MILP_Positive_Net_{scenario}"
        )

        increment_column = (
            f"MILP_Rounding_Increment_{scenario}"
        )

        rows.append(
            {
                "Scenario":
                    scenario,

                "Priority_Eligible_Count":
                    int(
                        eligible.sum()
                    ),

                "Positive_Decision_Row_Count":
                    int(
                        dataframe.loc[
                            eligible,
                            active_column,
                        ]
                        .sum()
                    ),

                "Whole_Unit_Gross_Requirement_Total":
                    int(
                        dataframe[
                            requirement_column
                        ]
                        .sum()
                    ),

                "Whole_Unit_Eligible_Net_Requirement_Total":
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

                "Whole_Unit_Rounding_Increment_Total":
                    float(
                        dataframe[
                            increment_column
                        ]
                        .sum()
                    ),

                "Rows_With_Positive_Rounding_Increment":
                    int(
                        dataframe[
                            increment_column
                        ]
                        .gt(
                            INTEGER_TOLERANCE
                        )
                        .sum()
                    ),

                "MILP_B100":
                    float(
                        dataframe.loc[
                            eligible,
                            value_column,
                        ]
                        .sum()
                    ),

                "Decision_Variable_Domain":
                    "NONNEGATIVE_INTEGER",

                "Decision_Unit":
                    "INVENTORY_REPORT_UNIT",

                "Scope":
                    "ELIGIBLE_ABC_RECORDS_ONLY",

                "Full_AB_Population_Represented":
                    False,

                "MILP_Input_Version":
                    MILP_INPUT_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BY ABC
# ============================================================

def build_by_abc(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    for scenario in SCENARIOS:

        net_column = (
            f"MILP_Net_Requirement_{scenario}"
        )

        value_column = (
            f"MILP_Full_Funding_Value_{scenario}"
        )

        active_column = (
            f"MILP_Positive_Net_{scenario}"
        )

        for abc_class in (
            PRIORITY_CLASSES
        ):

            mask = (
                eligible
                &
                dataframe[
                    "ABC_AsOf"
                ]
                .eq(
                    abc_class
                )
            )

            class_df = (
                dataframe.loc[
                    mask
                ]
            )

            rows.append(
                {
                    "Scenario":
                        scenario,

                    "ABC_AsOf":
                        abc_class,

                    "Eligible_Medicine_Count":
                        len(
                            class_df
                        ),

                    "Positive_Decision_Row_Count":
                        int(
                            class_df[
                                active_column
                            ]
                            .sum()
                        ),

                    "Net_Requirement_Units":
                        int(
                            class_df[
                                net_column
                            ]
                            .astype(
                                np.int64
                            )
                            .sum()
                        ),

                    "Full_Funding_Value":
                        float(
                            class_df[
                                value_column
                            ]
                            .sum()
                        ),

                    "MILP_Input_Version":
                        MILP_INPUT_VERSION,
                }
            )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# BUDGET SCENARIOS
# ============================================================

def build_budget_scenarios(
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, scenario_row in (
        scenario_summary.iterrows()
    ):

        b100 = float(
            scenario_row[
                "MILP_B100"
            ]
        )

        for factor in BUDGET_FACTORS:

            rows.append(
                {
                    "Requirement_Scenario":
                        scenario_row[
                            "Scenario"
                        ],

                    "Budget_Factor":
                        factor,

                    "Budget_Percent":
                        factor
                        * 100,

                    "MILP_B100":
                        b100,

                    "Scenario_Budget":
                        b100
                        * factor,

                    "Budget_Is_Actual_CDH_Budget":
                        False,

                    "Budget_Definition":
                        (
                            "Modeled percentage of "
                            "whole-unit eligible "
                            "full-funding requirement"
                        ),

                    "MILP_Input_Version":
                        MILP_INPUT_VERSION,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LEXICOGRAPHIC FORMULATION RECORD
# ============================================================

def build_formulation_record() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "MILP_Input_Version":
                    MILP_INPUT_VERSION,

                "Decision_Variable":
                    "x_i",

                "Decision_Domain":
                    (
                        "Nonnegative integer "
                        "inventory-report units"
                    ),

                "Decision_Bound":
                    "0 <= x_i <= R_i",

                "Net_Requirement":
                    "R_i = max(0, Q_i - S_i)",

                "Budget_Constraint":
                    "sum(C_i * x_i) <= B",

                "Stage_1":
                    (
                        "Maximize minimum "
                        "proportional coverage "
                        "theta_A among positive-net "
                        "Category A medicines."
                    ),

                "Stage_2":
                    (
                        "Preserve theta_A; maximize "
                        "minimum proportional "
                        "coverage theta_B among "
                        "positive-net Category B."
                    ),

                "Stage_3":
                    (
                        "Preserve theta_A and "
                        "theta_B; maximize minimum "
                        "proportional coverage "
                        "theta_C among positive-net "
                        "Category C."
                    ),

                "Stage_4":
                    (
                        "Preserve achieved A/B/C "
                        "coverage levels; maximize "
                        "total allocated procurement "
                        "value as deterministic "
                        "budget-use tie-break."
                    ),

                "Coverage_Constraint":
                    (
                        "x_i >= theta_class * R_i "
                        "for positive-net medicines "
                        "in that ABC class."
                    ),

                "Clinical_Priority":
                    False,

                "Supplier_Pack_Size_Constraints":
                    False,

                "Full_Hospital_Optimization_Claim":
                    False,

                "Reason_Full_Claim_Blocked":
                    (
                        "Five A/B medicines retain "
                        "unresolved baseline stock."
                    ),
            }
        ]
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance(
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "MILP_Input_Version":
                    MILP_INPUT_VERSION,

                "Target_Year":
                    2026,

                "Optimization_Scope":
                    "ELIGIBLE_ABC_RECORDS_ONLY",

                "Eligible_Medicine_Count":
                    int(
                        scenario_summary[
                            "Priority_Eligible_Count"
                        ]
                        .iloc[0]
                    ),

                "Decision_Variable_Integer":
                    True,

                "Whole_Unit_Reason":
                    (
                        "Validated stock and "
                        "point requirements are "
                        "recorded in whole "
                        "inventory-report units."
                    ),

                "P80_Rounding_Rule":
                    (
                        "ceil(max(incumbent, raw P80))"
                    ),

                "P90_Rounding_Rule":
                    (
                        "ceil(max(incumbent, raw P90))"
                    ),

                "Raw_P80_Changed":
                    False,

                "Raw_P90_Changed":
                    False,

                "Pack_Size_Data_Available":
                    False,

                "Pack_Multiple_Optimization":
                    False,

                "High_Priority_AB_Unresolved_Count":
                    5,

                "Full_High_Priority_Claim_Allowed":
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
    scenario_summary: pd.DataFrame,
    by_abc: pd.DataFrame,
    budgets: pd.DataFrame,
    formulation: pd.DataFrame,
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
            "scenario_df",
            scenario_summary,
        )

        connection.register(
            "abc_df",
            by_abc,
        )

        connection.register(
            "budget_df",
            budgets,
        )

        connection.register(
            "formulation_df",
            formulation,
        )

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_inputs AS
            SELECT *
            FROM inputs_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_scenario_summary AS
            SELECT *
            FROM scenario_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_by_abc AS
            SELECT *
            FROM abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_budget_scenarios AS
            SELECT *
            FROM budget_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_formulation AS
            SELECT *
            FROM formulation_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_governance AS
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
    print("2026 INTEGER-READY MILP INPUTS")
    print("=" * 112)
    print()

    source = (
        load_optimizer_inputs()
    )

    validate_source(
        source
    )

    print(
        "Optimizer scenario source "
        "validation: PASS"
    )

    inputs = (
        build_milp_inputs(
            source
        )
    )

    validate_milp_inputs(
        inputs
    )

    print(
        "MILP input structural "
        "validation: PASS"
    )

    scenario_summary = (
        build_scenario_summary(
            inputs
        )
    )

    by_abc = (
        build_by_abc(
            inputs
        )
    )

    budgets = (
        build_budget_scenarios(
            scenario_summary
        )
    )

    formulation = (
        build_formulation_record()
    )

    governance = (
        build_governance(
            scenario_summary
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    inputs.to_parquet(
        OUTPUT_DIR
        / "planning_2026_milp_inputs.parquet",
        index=False,
        engine="pyarrow",
    )

    inputs.to_csv(
        OUTPUT_DIR
        / "planning_2026_milp_inputs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scenario_summary.to_csv(
        OUTPUT_DIR
        / "planning_2026_milp_scenario_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "planning_2026_milp_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    budgets.to_csv(
        OUTPUT_DIR
        / "planning_2026_milp_budget_scenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )

    formulation.to_csv(
        OUTPUT_DIR
        / "planning_2026_milp_formulation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "planning_2026_milp_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        inputs=inputs,
        scenario_summary=scenario_summary,
        by_abc=by_abc,
        budgets=budgets,
        formulation=formulation,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 112)
    print("WHOLE-UNIT SCENARIO SUMMARY")
    print("-" * 112)

    print(
        scenario_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 112)
    print("WHOLE-UNIT REQUIREMENTS BY ABC")
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
    print("REVISED WHOLE-UNIT BUDGET SCENARIOS")
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
    print("LEXICOGRAPHIC MILP FORMULATION")
    print("-" * 112)

    print(
        formulation.to_string(
            index=False
        )
    )

    print()
    print("-" * 112)
    print("MILP GOVERNANCE")
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
        "Procurement decisions will be "
        "integer inventory-report units."
    )

    print(
        "P80/P90 whole-unit requirements "
        "are decision-policy quantities; "
        "the original statistical quantiles "
        "remain unchanged."
    )

    print(
        "Supplier pack sizes are not "
        "available, so no box/case multiple "
        "constraint is assumed."
    )

    print(
        "Zero-net-requirement medicines "
        "remain represented but require "
        "no procurement decision."
    )

    print(
        "No optimizer has been executed."
    )

    print()
    print("=" * 112)
    print(
        "INTEGER-READY MILP INPUT BUILD COMPLETE"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()