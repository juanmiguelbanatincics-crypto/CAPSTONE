from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import run_2025_retrospective_simulation as retro


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
    / "retrospective_2025_matched_budget_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

COMPARISON_VERSION = (
    "RETROSPECTIVE_2025_MATCHED_BUDGET_V1"
)

REFERENCE_SCENARIO = "POINT"

REFERENCE_BUDGET_FACTORS = [
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

SCENARIOS = retro.SCENARIOS

EXPECTED_ELIGIBLE = 2197

REPEATABILITY_SCENARIO = (
    "P80_FLOORED_CEIL"
)

REPEATABILITY_REFERENCE_FACTOR = 0.80

VALUE_TOLERANCE = 0.05


# ============================================================
# LOAD AUTHORITATIVE B100
# ============================================================

def load_b100() -> dict[str, float]:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            """
            SELECT
                Scenario,
                Retrospective_B100
            FROM retrospective_2025_scenario_summary
            ORDER BY Scenario
            """
        ).df()

    finally:

        connection.close()

    expected = set(
        SCENARIOS.keys()
    )

    observed = set(
        dataframe[
            "Scenario"
        ].tolist()
    )

    if observed != expected:

        raise ValueError(
            "Retrospective B100 scenario "
            "set does not match expected "
            "planning policies."
        )

    return {
        row.Scenario:
            float(
                row.Retrospective_B100
            )
        for row in dataframe.itertuples(
            index=False
        )
    }


# ============================================================
# BUILD MATCHED BUDGET GRID
# ============================================================

def build_budget_grid(
    b100: dict[str, float],
) -> pd.DataFrame:

    reference_b100 = (
        b100[
            REFERENCE_SCENARIO
        ]
    )

    rows = []

    for reference_factor in (
        REFERENCE_BUDGET_FACTORS
    ):

        common_budget = (
            reference_b100
            *
            reference_factor
        )

        for scenario in SCENARIOS:

            scenario_b100 = (
                b100[
                    scenario
                ]
            )

            scenario_factor = (
                common_budget
                /
                scenario_b100
            )

            if (
                scenario_factor
                >
                1.0
                +
                1e-9
            ):

                raise ValueError(
                    "Matched budget exceeds "
                    f"{scenario} B100."
                )

            rows.append(
                {
                    "Reference_Scenario":
                        REFERENCE_SCENARIO,

                    "Reference_B100":
                        reference_b100,

                    "Reference_Budget_Factor":
                        reference_factor,

                    "Reference_Budget_Percent":
                        (
                            reference_factor
                            *
                            100
                        ),

                    "Matched_Peso_Budget":
                        common_budget,

                    "Requirement_Scenario":
                        scenario,

                    "Scenario_B100":
                        scenario_b100,

                    "Scenario_Specific_Budget_Factor":
                        scenario_factor,

                    "Scenario_Specific_Budget_Percent":
                        (
                            scenario_factor
                            *
                            100
                        ),

                    "Comparison_Version":
                        COMPARISON_VERSION,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ADD ECONOMIC EVALUATION
# ============================================================

def add_economic_outcomes(
    allocation: pd.DataFrame,
) -> pd.DataFrame:

    result = allocation.copy()

    result[
        "Actual_2025_Issuance_Value_At_Planning_Cost"
    ] = (
        result[
            "Actual_2025_Annual_Issuance"
        ]
        *
        result[
            "Planning_Cost"
        ]
    )

    result[
        "Simulated_Covered_Issuance_Value_At_Planning_Cost"
    ] = (
        result[
            "Simulated_Issuance_Covered_Units"
        ]
        *
        result[
            "Planning_Cost"
        ]
    )

    result[
        "Comparison_Version"
    ] = COMPARISON_VERSION

    return result


# ============================================================
# SUMMARIZE ONE MATCHED CASE
# ============================================================

def summarize_case(
    allocation: pd.DataFrame,
    base_summary: dict,
    reference_factor: float,
    matched_budget: float,
    scenario_factor: float,
) -> dict:

    actual_value = float(
        allocation[
            "Actual_2025_Issuance_Value_At_Planning_Cost"
        ].sum()
    )

    covered_value = float(
        allocation[
            "Simulated_Covered_Issuance_Value_At_Planning_Cost"
        ].sum()
    )

    shortfall_value = float(
        allocation[
            "Simulated_Shortfall_Value_At_Planning_Cost"
        ].sum()
    )

    residual_value = float(
        allocation[
            "Simulated_Residual_Value_At_Planning_Cost"
        ].sum()
    )

    stock_only_shortfall = int(
        allocation[
            "Stock_Only_Simulated_Shortfall_Units"
        ].sum()
    )

    simulated_shortfall = int(
        allocation[
            "Simulated_Annual_Issuance_Shortfall_Units"
        ].sum()
    )

    return {
        "Requirement_Scenario":
            base_summary[
                "Requirement_Scenario"
            ],

        "Reference_Budget_Percent":
            (
                reference_factor
                *
                100
            ),

        "Matched_Peso_Budget":
            matched_budget,

        "Scenario_B100":
            base_summary[
                "B100"
            ],

        "Scenario_Specific_Budget_Factor":
            scenario_factor,

        "Scenario_Specific_Budget_Percent":
            (
                scenario_factor
                *
                100
            ),

        "Allocated_Value":
            float(
                base_summary[
                    "Allocated_Value"
                ]
            ),

        "Unspent_Budget":
            float(
                matched_budget
                -
                base_summary[
                    "Allocated_Value"
                ]
            ),

        "Budget_Utilization_Pct":
            (
                float(
                    base_summary[
                        "Allocated_Value"
                    ]
                )
                /
                matched_budget
                *
                100
                if matched_budget > 0
                else 100.0
            ),

        "Actual_2025_Issuance_Eligible_Units":
            int(
                base_summary[
                    "Actual_2025_Issuance_Eligible_Units"
                ]
            ),

        "Simulated_Covered_Issuance_Units":
            int(
                base_summary[
                    "Simulated_Issuance_Covered_Units"
                ]
            ),

        "Simulated_Annual_Issuance_Shortfall_Units":
            simulated_shortfall,

        "Simulated_Residual_Inventory_Units":
            int(
                base_summary[
                    "Simulated_Residual_Inventory_Units"
                ]
            ),

        "Stock_Only_Simulated_Shortfall_Units":
            stock_only_shortfall,

        "Shortfall_Reduction_vs_Stock_Only_Units":
            (
                stock_only_shortfall
                -
                simulated_shortfall
            ),

        "Simulated_Fulfillment_Pct":
            float(
                base_summary[
                    "Simulated_Fulfillment_Pct"
                ]
            ),

        "Actual_2025_Issuance_Value_At_Planning_Cost":
            actual_value,

        "Simulated_Covered_Issuance_Value_At_Planning_Cost":
            covered_value,

        "Simulated_Shortfall_Value_At_Planning_Cost":
            shortfall_value,

        "Simulated_Residual_Value_At_Planning_Cost":
            residual_value,

        "Economic_Fulfillment_Pct":
            (
                covered_value
                /
                actual_value
                *
                100
                if actual_value > 0
                else 100.0
            ),

        "Simulated_Shortfall_Medicine_Count":
            int(
                base_summary[
                    "Simulated_Shortfall_Medicine_Count"
                ]
            ),

        "Simulated_Residual_Medicine_Count":
            int(
                base_summary[
                    "Simulated_Residual_Medicine_Count"
                ]
            ),

        "Observed_Min_A":
            float(
                base_summary[
                    "Observed_Min_A"
                ]
            ),

        "Observed_Min_B":
            float(
                base_summary[
                    "Observed_Min_B"
                ]
            ),

        "Observed_Min_C":
            float(
                base_summary[
                    "Observed_Min_C"
                ]
            ),

        "Actual_2025_Used_In_Optimization":
            False,

        "Comparison_Version":
            COMPARISON_VERSION,
    }


# ============================================================
# BY ABC
# ============================================================

def build_by_abc(
    allocations: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouping = allocations.groupby(
        [
            "Requirement_Scenario",
            "Reference_Budget_Percent",
            "ABC_AsOf",
        ],
        sort=True,
    )

    for (
        scenario,
        reference_budget,
        abc_class,
    ), group in grouping:

        actual_units = int(
            group[
                "Actual_2025_Annual_Issuance"
            ].sum()
        )

        covered_units = int(
            group[
                "Simulated_Issuance_Covered_Units"
            ].sum()
        )

        actual_value = float(
            group[
                "Actual_2025_Issuance_Value_At_Planning_Cost"
            ].sum()
        )

        covered_value = float(
            group[
                "Simulated_Covered_Issuance_Value_At_Planning_Cost"
            ].sum()
        )

        rows.append(
            {
                "Requirement_Scenario":
                    scenario,

                "Reference_Budget_Percent":
                    reference_budget,

                "Matched_Peso_Budget":
                    float(
                        group[
                            "Matched_Peso_Budget"
                        ].iloc[0]
                    ),

                "ABC_AsOf":
                    abc_class,

                "Actual_2025_Issuance_Units":
                    actual_units,

                "Simulated_Covered_Issuance_Units":
                    covered_units,

                "Simulated_Shortfall_Units":
                    int(
                        group[
                            "Simulated_Annual_Issuance_Shortfall_Units"
                        ].sum()
                    ),

                "Simulated_Residual_Units":
                    int(
                        group[
                            "Simulated_Residual_Inventory_Units"
                        ].sum()
                    ),

                "Unit_Fulfillment_Pct":
                    (
                        covered_units
                        /
                        actual_units
                        *
                        100
                        if actual_units > 0
                        else 100.0
                    ),

                "Actual_Issuance_Value_At_Planning_Cost":
                    actual_value,

                "Covered_Issuance_Value_At_Planning_Cost":
                    covered_value,

                "Shortfall_Value_At_Planning_Cost":
                    float(
                        group[
                            "Simulated_Shortfall_Value_At_Planning_Cost"
                        ].sum()
                    ),

                "Residual_Value_At_Planning_Cost":
                    float(
                        group[
                            "Simulated_Residual_Value_At_Planning_Cost"
                        ].sum()
                    ),

                "Economic_Fulfillment_Pct":
                    (
                        covered_value
                        /
                        actual_value
                        *
                        100
                        if actual_value > 0
                        else 100.0
                    ),

                "Comparison_Version":
                    COMPARISON_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PARETO / TRADE-OFF AUDIT
# ============================================================

def build_tradeoff_audit(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

    for reference_budget, group in (
        summary.groupby(
            "Reference_Budget_Percent",
            sort=True,
        )
    ):

        current = group.copy()

        current[
            "Shortfall_Value_Rank"
        ] = (
            current[
                "Simulated_Shortfall_Value_At_Planning_Cost"
            ]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

        current[
            "Residual_Value_Rank"
        ] = (
            current[
                "Simulated_Residual_Value_At_Planning_Cost"
            ]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

        dominated_flags = []

        for _, row in current.iterrows():

            dominated = False

            for _, other in current.iterrows():

                if (
                    other[
                        "Requirement_Scenario"
                    ]
                    ==
                    row[
                        "Requirement_Scenario"
                    ]
                ):
                    continue

                no_worse_shortfall = (
                    other[
                        "Simulated_Shortfall_Value_At_Planning_Cost"
                    ]
                    <=
                    row[
                        "Simulated_Shortfall_Value_At_Planning_Cost"
                    ]
                    +
                    VALUE_TOLERANCE
                )

                no_worse_residual = (
                    other[
                        "Simulated_Residual_Value_At_Planning_Cost"
                    ]
                    <=
                    row[
                        "Simulated_Residual_Value_At_Planning_Cost"
                    ]
                    +
                    VALUE_TOLERANCE
                )

                strictly_better_one = (
                    other[
                        "Simulated_Shortfall_Value_At_Planning_Cost"
                    ]
                    <
                    row[
                        "Simulated_Shortfall_Value_At_Planning_Cost"
                    ]
                    -
                    VALUE_TOLERANCE
                    or
                    other[
                        "Simulated_Residual_Value_At_Planning_Cost"
                    ]
                    <
                    row[
                        "Simulated_Residual_Value_At_Planning_Cost"
                    ]
                    -
                    VALUE_TOLERANCE
                )

                if (
                    no_worse_shortfall
                    and
                    no_worse_residual
                    and
                    strictly_better_one
                ):

                    dominated = True
                    break

            dominated_flags.append(
                dominated
            )

        current[
            "Pareto_Dominated_On_Shortfall_Residual_Value"
        ] = dominated_flags

        frames.append(
            current[
                [
                    "Requirement_Scenario",
                    "Reference_Budget_Percent",
                    "Matched_Peso_Budget",
                    "Simulated_Shortfall_Value_At_Planning_Cost",
                    "Simulated_Residual_Value_At_Planning_Cost",
                    "Economic_Fulfillment_Pct",
                    "Shortfall_Value_Rank",
                    "Residual_Value_Rank",
                    "Pareto_Dominated_On_Shortfall_Residual_Value",
                    "Comparison_Version",
                ]
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# REPEATABILITY
# ============================================================

def allocation_hash(
    dataframe: pd.DataFrame,
) -> str:

    ordered = dataframe[
        [
            "Medicine_Key",
            "Allocated_Units",
        ]
    ].sort_values(
        "Medicine_Key"
    )

    payload = "\n".join(
        f"{row.Medicine_Key}:"
        f"{int(row.Allocated_Units)}"
        for row
        in ordered.itertuples(
            index=False
        )
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def run_repeatability(
    eligible: pd.DataFrame,
    b100: dict[str, float],
    first_solution: pd.DataFrame,
) -> pd.DataFrame:

    common_budget = (
        b100[
            REFERENCE_SCENARIO
        ]
        *
        REPEATABILITY_REFERENCE_FACTOR
    )

    scenario_factor = (
        common_budget
        /
        b100[
            REPEATABILITY_SCENARIO
        ]
    )

    second, second_summary = (
        retro.solve_case(
            eligible_data=eligible,
            scenario=(
                REPEATABILITY_SCENARIO
            ),
            requirement_column=(
                SCENARIOS[
                    REPEATABILITY_SCENARIO
                ]
            ),
            budget_factor=(
                scenario_factor
            ),
            run_technical_tiebreak=False,
            deterministic_solver=True,
        )
    )

    retro.validate_case(
        second,
        second_summary,
    )

    first = (
        first_solution
        .sort_values(
            "Medicine_Key"
        )
        .reset_index(
            drop=True
        )
    )

    second = (
        second
        .sort_values(
            "Medicine_Key"
        )
        .reset_index(
            drop=True
        )
    )

    mismatches = int(
        (
            first[
                "Allocated_Units"
            ]
            .to_numpy()
            !=
            second[
                "Allocated_Units"
            ]
            .to_numpy()
        )
        .sum()
    )

    first_hash = allocation_hash(
        first
    )

    second_hash = allocation_hash(
        second
    )

    passed = bool(
        mismatches == 0
        and
        first_hash == second_hash
    )

    if not passed:

        raise RuntimeError(
            "Matched-budget repeatability "
            "validation failed."
        )

    return pd.DataFrame(
        [
            {
                "Requirement_Scenario":
                    REPEATABILITY_SCENARIO,

                "Reference_Budget_Percent":
                    (
                        REPEATABILITY_REFERENCE_FACTOR
                        *
                        100
                    ),

                "Matched_Peso_Budget":
                    common_budget,

                "Scenario_Specific_Budget_Percent":
                    (
                        scenario_factor
                        *
                        100
                    ),

                "Allocation_Mismatch_Count":
                    mismatches,

                "First_SHA256":
                    first_hash,

                "Second_SHA256":
                    second_hash,

                "Exact_Repeatability_PASS":
                    passed,

                "Comparison_Version":
                    COMPARISON_VERSION,
            }
        ]
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance(
    reference_b100: float,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "Comparison_Version":
                    COMPARISON_VERSION,

                "Planning_Origin":
                    "END_OF_2024",

                "Target_Year":
                    2025,

                "Common_Budget_Reference":
                    "POINT_RETROSPECTIVE_B100",

                "Reference_B100":
                    reference_b100,

                "Matched_Budget_Levels":
                    (
                        "60%,70%,80%,90%,100% "
                        "of POINT B100"
                    ),

                "Same_Peso_Budget_Across_Policies":
                    True,

                "Optimizer_Policy":
                    "FROZEN_STRICT_ABC_V2",

                "Substantive_Lexicographic_Stages":
                    "STAGES_1_TO_6",

                "Technical_Medicine_Key_Tie_Break_Run":
                    False,

                "Deterministic_Solver_Configuration":
                    "parallel=off; threads=1; random_seed=0",

                "Repeatability_Audit_Required":
                    True,
                "Actual_2025_Entered_Optimization":
                    False,

                "Actual_2025_Used_For_Evaluation":
                    True,

                "Primary_Cross_Policy_Metrics":
                    (
                        "Shortfall value, residual "
                        "value, economic fulfillment"
                    ),

                "Raw_Unit_Totals_Used_As_Primary_"
                "Cross_Medicine_Economic_Metric":
                    False,

                "Actual_Stockout_Claim":
                    False,

                "Actual_Waste_Claim":
                    False,

                "Matched_Budgets_Are_Actual_CDH_Budgets":
                    False,

                "Full_Hospital_Claim":
                    False,

                "Reason_Full_Claim_Blocked":
                    (
                        "Four A/B medicines were "
                        "unresolved at the "
                        "end-of-2024 planning origin."
                    ),
            }
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    budget_grid,
    allocations,
    summary,
    by_abc,
    tradeoff,
    repeatability,
    governance,
):

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "grid_df",
            budget_grid,
        )

        connection.register(
            "alloc_df",
            allocations,
        )

        connection.register(
            "summary_df",
            summary,
        )

        connection.register(
            "abc_df",
            by_abc,
        )

        connection.register(
            "tradeoff_df",
            tradeoff,
        )

        connection.register(
            "repeat_df",
            repeatability,
        )

        connection.register(
            "gov_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_grid AS
            SELECT * FROM grid_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_allocations AS
            SELECT * FROM alloc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_summary AS
            SELECT * FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_by_abc AS
            SELECT * FROM abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_tradeoff AS
            SELECT * FROM tradeoff_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_repeatability AS
            SELECT * FROM repeat_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_governance AS
            SELECT * FROM gov_df
            """
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 120)
    print("CDH INVENTORY DSS")
    print("2025 RETROSPECTIVE MATCHED-PESO POLICY COMPARISON")
    print("=" * 120)
    print()

    full_data = retro.load_inputs()

    retro.validate_inputs(
        full_data
    )

    print(
        "Retrospective temporal "
        "validation: PASS"
    )

    eligible = full_data.loc[
        full_data[
            "Retrospective_Optimization_Eligible"
        ]
    ].copy()

    if len(eligible) != EXPECTED_ELIGIBLE:

        raise ValueError(
            "Unexpected eligible population."
        )

    b100 = load_b100()

    budget_grid = build_budget_grid(
        b100
    )

    reference_b100 = (
        b100[
            REFERENCE_SCENARIO
        ]
    )

    print(
        "Eligible medicines:",
        f"{len(eligible):,}",
    )

    print(
        "Reference policy:",
        REFERENCE_SCENARIO,
    )

    print(
        "Reference B100:",
        f"{reference_b100:,.2f}",
    )

    print(
        "Matched budgets:",
        "60%, 70%, 80%, 90%, 100% "
        "of POINT B100",
    )

    print(
        "Policies compared:",
        ", ".join(
            SCENARIOS.keys()
        ),
    )

    print(
        "Actual 2025 issuance enters "
        "optimization model: NO"
    )

    print()

    allocation_frames = []
    summary_rows = []

    for reference_factor in (
        REFERENCE_BUDGET_FACTORS
    ):

        matched_budget = (
            reference_b100
            *
            reference_factor
        )

        for scenario, requirement_column in (
            SCENARIOS.items()
        ):

            scenario_factor = (
                matched_budget
                /
                b100[
                    scenario
                ]
            )

            print("-" * 120)

            print(
                "Matched-peso solve:",
                scenario,
            )

            print(
                "  Common budget:",
                f"{matched_budget:,.2f}",
            )

            print(
                "  Common reference level:",
                f"{reference_factor:.0%} "
                "of POINT B100",
            )

            print(
                "  Scenario-specific "
                "B100 percentage:",
                f"{scenario_factor * 100:,.4f}%",
            )

            allocation, base_summary = (
                retro.solve_case(
                    eligible_data=eligible,
                    scenario=scenario,
                    requirement_column=(
                        requirement_column
                    ),
                    budget_factor=(
                        scenario_factor
                    ),
                    run_technical_tiebreak=False,
                    deterministic_solver=True,
                )
            )

            retro.validate_case(
                allocation,
                base_summary,
            )

            if abs(
                float(
                    base_summary[
                        "Scenario_Budget"
                    ]
                )
                -
                matched_budget
            ) > VALUE_TOLERANCE:

                raise RuntimeError(
                    "Matched peso budget "
                    "was not reproduced."
                )

            allocation = add_economic_outcomes(
                allocation
            )

            allocation[
                "Reference_Budget_Factor"
            ] = reference_factor

            allocation[
                "Reference_Budget_Percent"
            ] = (
                reference_factor
                *
                100
            )

            allocation[
                "Matched_Peso_Budget"
            ] = matched_budget

            allocation[
                "Scenario_Specific_Budget_Factor"
            ] = scenario_factor

            allocation[
                "Scenario_Specific_Budget_Percent"
            ] = (
                scenario_factor
                *
                100
            )

            allocation_frames.append(
                allocation
            )

            row = summarize_case(
                allocation=allocation,
                base_summary=base_summary,
                reference_factor=(
                    reference_factor
                ),
                matched_budget=(
                    matched_budget
                ),
                scenario_factor=(
                    scenario_factor
                ),
            )

            summary_rows.append(
                row
            )

            print(
                "  Simulated shortfall:",
                f"{row['Simulated_Annual_Issuance_Shortfall_Units']:,}",
            )

            print(
                "  Simulated residual:",
                f"{row['Simulated_Residual_Inventory_Units']:,}",
            )

            print(
                "  Shortfall value:",
                f"{row['Simulated_Shortfall_Value_At_Planning_Cost']:,.2f}",
            )

            print(
                "  Residual value:",
                f"{row['Simulated_Residual_Value_At_Planning_Cost']:,.2f}",
            )

            print(
                "  Economic fulfillment:",
                f"{row['Economic_Fulfillment_Pct']:,.4f}%",
            )

    allocations = pd.concat(
        allocation_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    by_abc = build_by_abc(
        allocations
    )

    tradeoff = build_tradeoff_audit(
        summary
    )

    first_repeat = allocations.loc[
        allocations[
            "Requirement_Scenario"
        ].eq(
            REPEATABILITY_SCENARIO
        )
        &
        np.isclose(
            allocations[
                "Reference_Budget_Factor"
            ],
            REPEATABILITY_REFERENCE_FACTOR,
        )
    ].copy()

    print()
    print(
        "Running matched-budget "
        "P80 @ POINT-80%-budget "
        "repeatability check..."
    )

    repeatability = run_repeatability(
        eligible=eligible,
        b100=b100,
        first_solution=first_repeat,
    )

    print(
        "Matched-budget repeatability: PASS"
    )

    governance = build_governance(
        reference_b100
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    budget_grid.to_csv(
        OUTPUT_DIR
        / "matched_budget_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )

    allocations.to_parquet(
        OUTPUT_DIR
        / "matched_budget_allocations.parquet",
        index=False,
        engine="pyarrow",
    )

    allocations.to_csv(
        OUTPUT_DIR
        / "matched_budget_allocations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "matched_budget_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "matched_budget_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tradeoff.to_csv(
        OUTPUT_DIR
        / "matched_budget_tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeatability.to_csv(
        OUTPUT_DIR
        / "matched_budget_repeatability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "matched_budget_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        budget_grid=budget_grid,
        allocations=allocations,
        summary=summary,
        by_abc=by_abc,
        tradeoff=tradeoff,
        repeatability=repeatability,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 120)
    print("MATCHED-PESO POLICY SUMMARY")
    print("=" * 120)

    display_columns = [
        "Reference_Budget_Percent",
        "Matched_Peso_Budget",
        "Requirement_Scenario",
        "Scenario_Specific_Budget_Percent",
        "Allocated_Value",
        "Simulated_Annual_Issuance_Shortfall_Units",
        "Simulated_Residual_Inventory_Units",
        "Simulated_Fulfillment_Pct",
        "Simulated_Shortfall_Value_At_Planning_Cost",
        "Simulated_Residual_Value_At_Planning_Cost",
        "Economic_Fulfillment_Pct",
        "Simulated_Shortfall_Medicine_Count",
        "Simulated_Residual_Medicine_Count",
    ]

    print(
        summary[
            display_columns
        ]
        .sort_values(
            [
                "Reference_Budget_Percent",
                "Requirement_Scenario",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("=" * 120)
    print("MATCHED-PESO TRADE-OFF AUDIT")
    print("=" * 120)

    print(
        tradeoff.sort_values(
            [
                "Reference_Budget_Percent",
                "Requirement_Scenario",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("=" * 120)
    print("MATCHED-PESO BY AS-OF ABC")
    print("=" * 120)

    print(
        by_abc.sort_values(
            [
                "Reference_Budget_Percent",
                "Requirement_Scenario",
                "ABC_AsOf",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("=" * 120)
    print("REPEATABILITY")
    print("=" * 120)

    print(
        repeatability.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("GOVERNANCE")
    print("=" * 120)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 120)
    print("INTERPRETATION")
    print("-" * 120)

    print(
        "All three requirement policies "
        "receive the same peso amount "
        "within each comparison level."
    )

    print(
        "Therefore cross-policy differences "
        "at a given matched budget are not "
        "caused by one policy receiving "
        "more modeled money."
    )

    print(
        "Shortfall and residual values at "
        "planning cost are the primary "
        "cross-policy economic comparison "
        "metrics."
    )

    print(
        "Aggregated unit totals remain "
        "secondary because medicine units "
        "are heterogeneous."
    )

    print(
        "A policy is marked Pareto-dominated "
        "only if another policy has no higher "
        "shortfall value and no higher "
        "residual value, with at least one "
        "strict improvement."
    )

    print(
        "This analysis does not declare a "
        "single universally optimal policy "
        "when shortfall and residual trade off."
    )

    print(
        "Actual 2025 issuance was revealed "
        "only after each allocation was fixed."
    )

    print(
        "Results are simulated annual outcomes, "
        "not actual stockouts or waste."
    )

    print()
    print("=" * 120)
    print(
        "2025 MATCHED-PESO COMPARISON COMPLETE"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
