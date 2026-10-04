from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import highspy
import numpy as np
import pandas as pd
import pyomo

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    NonNegativeIntegers,
    Objective,
    Param,
    Set,
    SolverFactory,
    UnitInterval,
    Var,
    maximize,
    value,
)

from pyomo.opt import TerminationCondition


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
    / "milp_budget_optimization_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

OPTIMIZATION_VERSION = (
    "MILP_BUDGET_OPTIMIZATION_2026_V1"
)

SCENARIOS = {
    "POINT":
        "MILP_Net_Requirement_POINT",

    "P80_FLOORED_CEIL":
        "MILP_Net_Requirement_P80_FLOORED_CEIL",

    "P90_FLOORED_CEIL":
        "MILP_Net_Requirement_P90_FLOORED_CEIL",
}

BUDGET_FACTORS = [
    0.60,
    0.70,
    0.80,
    0.90,
]

ABC_CLASSES = [
    "A",
    "B",
    "C",
]

EXPECTED_ELIGIBLE = 2365

THETA_PRESERVE_TOLERANCE = 1e-7
VALUE_PRESERVE_TOLERANCE = 0.01
BUDGET_TOLERANCE = 0.05
COVERAGE_TOLERANCE = 1e-5

REPEATABILITY_SCENARIO = "POINT"
REPEATABILITY_BUDGET_FACTOR = 0.80


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            """
            SELECT *
            FROM planning_2026_milp_inputs
            WHERE Optimization_Priority_Eligible = TRUE
            ORDER BY ABC_Priority_Order, Medicine_Key
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

    if len(dataframe) != EXPECTED_ELIGIBLE:

        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE:,} "
            f"eligible rows; found "
            f"{len(dataframe):,}"
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

    if not (
        dataframe[
            "ABC_AsOf"
        ]
        .isin(
            ABC_CLASSES
        )
        .all()
    ):

        raise ValueError(
            "Non-A/B/C record entered "
            "the MILP population."
        )

    if (
        dataframe[
            "Planning_Cost"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Missing planning cost."
        )

    if (
        dataframe[
            "Planning_Cost"
        ]
        .le(0)
        .any()
    ):

        raise ValueError(
            "Non-positive planning cost."
        )

    for requirement_column in (
        SCENARIOS.values()
    ):

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

        values = (
            dataframe[
                requirement_column
            ]
            .astype(float)
            .to_numpy()
        )

        if (
            values < 0
        ).any():

            raise ValueError(
                f"Negative {requirement_column}."
            )

        if not np.allclose(
            values,
            np.round(
                values
            ),
            atol=1e-9,
            rtol=0,
        ):

            raise ValueError(
                f"{requirement_column} "
                "is not whole-unit."
            )


# ============================================================
# BUILD ONE MILP
# ============================================================

def build_model(
    dataframe: pd.DataFrame,
    requirement_column: str,
    budget_factor: float,
) -> tuple[
    ConcreteModel,
    float,
    float,
]:

    medicine_keys = (
        dataframe[
            "Medicine_Key"
        ]
        .tolist()
    )

    cost = dict(
        zip(
            dataframe[
                "Medicine_Key"
            ],
            dataframe[
                "Planning_Cost"
            ]
            .astype(float),
        )
    )

    requirement = dict(
        zip(
            dataframe[
                "Medicine_Key"
            ],
            dataframe[
                requirement_column
            ]
            .astype(int),
        )
    )

    abc = dict(
        zip(
            dataframe[
                "Medicine_Key"
            ],
            dataframe[
                "ABC_AsOf"
            ],
        )
    )

    # Stable technical rank.
    # Lower Medicine_Key receives
    # larger tie-break weight.
    sorted_keys = sorted(
        medicine_keys
    )

    tie_weight = {
        key:
            len(sorted_keys)
            -
            rank
        for rank, key in enumerate(
            sorted_keys
        )
    }

    b100 = float(
        sum(
            cost[key]
            *
            requirement[key]
            for key in medicine_keys
        )
    )

    budget = float(
        b100
        *
        budget_factor
    )

    model = ConcreteModel()

    model.I = Set(
        initialize=medicine_keys,
        ordered=True,
    )

    model.R = Param(
        model.I,
        initialize=requirement,
        within=NonNegativeIntegers,
    )

    model.C = Param(
        model.I,
        initialize=cost,
    )

    model.W = Param(
        model.I,
        initialize=tie_weight,
        within=NonNegativeIntegers,
    )

    model.x = Var(
        model.I,
        domain=NonNegativeIntegers,
        bounds=lambda m, i: (
            0,
            int(
                value(
                    m.R[i]
                )
            ),
        ),
    )

    # --------------------------------------------------------
    # BUDGET
    # --------------------------------------------------------

    model.total_value = Expression(
        expr=sum(
            model.C[i]
            *
            model.x[i]
            for i in model.I
        )
    )

    model.budget = Constraint(
        expr=(
            model.total_value
            <=
            budget
        )
    )

    # --------------------------------------------------------
    # CLASS COVERAGE
    # --------------------------------------------------------

    model.theta_A = Var(
        domain=UnitInterval
    )

    model.theta_B = Var(
        domain=UnitInterval
    )

    model.theta_C = Var(
        domain=UnitInterval
    )

    positive_by_class = {
        abc_class: [
            key
            for key in medicine_keys
            if (
                abc[key]
                ==
                abc_class
                and
                requirement[key]
                >
                0
            )
        ]
        for abc_class
        in ABC_CLASSES
    }

    if not positive_by_class["A"]:
        model.theta_A.fix(1.0)

    if not positive_by_class["B"]:
        model.theta_B.fix(1.0)

    if not positive_by_class["C"]:
        model.theta_C.fix(1.0)

    model.A_POSITIVE = Set(
        initialize=(
            positive_by_class["A"]
        )
    )

    model.B_POSITIVE = Set(
        initialize=(
            positive_by_class["B"]
        )
    )

    model.C_POSITIVE = Set(
        initialize=(
            positive_by_class["C"]
        )
    )

    def coverage_a_rule(
        m,
        i,
    ):

        return (
            m.x[i]
            >=
            m.theta_A
            *
            m.R[i]
        )

    def coverage_b_rule(
        m,
        i,
    ):

        return (
            m.x[i]
            >=
            m.theta_B
            *
            m.R[i]
        )

    def coverage_c_rule(
        m,
        i,
    ):

        return (
            m.x[i]
            >=
            m.theta_C
            *
            m.R[i]
        )

    model.coverage_A = Constraint(
        model.A_POSITIVE,
        rule=coverage_a_rule,
    )

    model.coverage_B = Constraint(
        model.B_POSITIVE,
        rule=coverage_b_rule,
    )

    model.coverage_C = Constraint(
        model.C_POSITIVE,
        rule=coverage_c_rule,
    )

    # --------------------------------------------------------
    # TECHNICAL TIE-BREAK SCORE
    # --------------------------------------------------------

    model.tie_score = Expression(
        expr=sum(
            model.W[i]
            *
            model.x[i]
            for i in model.I
        )
    )

    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    model.objective_A = Objective(
        expr=model.theta_A,
        sense=maximize,
    )

    model.objective_B = Objective(
        expr=model.theta_B,
        sense=maximize,
    )

    model.objective_C = Objective(
        expr=model.theta_C,
        sense=maximize,
    )

    model.objective_value = Objective(
        expr=model.total_value,
        sense=maximize,
    )

    model.objective_tie = Objective(
        expr=model.tie_score,
        sense=maximize,
    )

    model.objective_B.deactivate()
    model.objective_C.deactivate()
    model.objective_value.deactivate()
    model.objective_tie.deactivate()

    return (
        model,
        b100,
        budget,
    )


# ============================================================
# SOLVER HELPER
# ============================================================

def make_solver():

    solver = SolverFactory(
        "appsi_highs"
    )

    if not solver.available():

        raise RuntimeError(
            "appsi_highs unavailable."
        )

    return solver


def solve_and_check(
    solver,
    model: ConcreteModel,
    stage_name: str,
) -> None:

    results = solver.solve(
        model
    )

    termination = (
        results.solver
        .termination_condition
    )

    if (
        termination
        !=
        TerminationCondition.optimal
    ):

        raise RuntimeError(
            f"{stage_name}: "
            f"termination = "
            f"{termination}"
        )


# ============================================================
# LEXICOGRAPHIC SOLVE
# ============================================================

def solve_budget_case(
    dataframe: pd.DataFrame,
    scenario: str,
    requirement_column: str,
    budget_factor: float,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    model, b100, budget = (
        build_model(
            dataframe=dataframe,
            requirement_column=(
                requirement_column
            ),
            budget_factor=(
                budget_factor
            ),
        )
    )

    solver = make_solver()

    # --------------------------------------------------------
    # STAGE 1 — A
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        (
            f"{scenario} "
            f"{budget_factor:.0%} "
            "Stage 1 A"
        ),
    )

    stage1_theta_a = float(
        value(
            model.theta_A
        )
    )

    model.preserve_A = Constraint(
        expr=(
            model.theta_A
            >=
            max(
                0.0,
                stage1_theta_a
                -
                THETA_PRESERVE_TOLERANCE,
            )
        )
    )

    model.objective_A.deactivate()
    model.objective_B.activate()

    # --------------------------------------------------------
    # STAGE 2 — B
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        (
            f"{scenario} "
            f"{budget_factor:.0%} "
            "Stage 2 B"
        ),
    )

    stage2_theta_b = float(
        value(
            model.theta_B
        )
    )

    model.preserve_B = Constraint(
        expr=(
            model.theta_B
            >=
            max(
                0.0,
                stage2_theta_b
                -
                THETA_PRESERVE_TOLERANCE,
            )
        )
    )

    model.objective_B.deactivate()
    model.objective_C.activate()

    # --------------------------------------------------------
    # STAGE 3 — C
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        (
            f"{scenario} "
            f"{budget_factor:.0%} "
            "Stage 3 C"
        ),
    )

    stage3_theta_c = float(
        value(
            model.theta_C
        )
    )

    model.preserve_C = Constraint(
        expr=(
            model.theta_C
            >=
            max(
                0.0,
                stage3_theta_c
                -
                THETA_PRESERVE_TOLERANCE,
            )
        )
    )

    model.objective_C.deactivate()
    model.objective_value.activate()

    # --------------------------------------------------------
    # STAGE 4 — MAXIMUM BUDGET USE
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        (
            f"{scenario} "
            f"{budget_factor:.0%} "
            "Stage 4 value"
        ),
    )

    stage4_value = float(
        value(
            model.total_value
        )
    )

    model.preserve_value = Constraint(
        expr=(
            model.total_value
            >=
            stage4_value
            -
            VALUE_PRESERVE_TOLERANCE
        )
    )

    model.objective_value.deactivate()
    model.objective_tie.activate()

    # --------------------------------------------------------
    # STAGE 5 — TECHNICAL DETERMINISTIC TIE-BREAK
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        (
            f"{scenario} "
            f"{budget_factor:.0%} "
            "Stage 5 tie-break"
        ),
    )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    allocation = dataframe[
        [
            "Medicine_Key",
            "Brand_Name",
            "Generic_Name",
            "Unit",
            "ABC_AsOf",
            "Behavior_Segment_AsOf",
            "Planning_Cost",
        ]
    ].copy()

    allocation[
        "Requirement_Scenario"
    ] = scenario

    allocation[
        "Budget_Factor"
    ] = budget_factor

    allocation[
        "Budget_Percent"
    ] = (
        budget_factor
        *
        100
    )

    allocation[
        "B100"
    ] = b100

    allocation[
        "Scenario_Budget"
    ] = budget

    allocation[
        "Net_Requirement"
    ] = (
        dataframe[
            requirement_column
        ]
        .astype(
            np.int64
        )
        .to_numpy()
    )

    allocation[
        "Allocated_Units"
    ] = [
        int(
            round(
                value(
                    model.x[key]
                )
            )
        )
        for key in dataframe[
            "Medicine_Key"
        ]
    ]

    allocation[
        "Unfunded_Requirement_Units"
    ] = (
        allocation[
            "Net_Requirement"
        ]
        -
        allocation[
            "Allocated_Units"
        ]
    )

    allocation[
        "Item_Coverage_Ratio"
    ] = np.where(
        allocation[
            "Net_Requirement"
        ]
        .gt(0),
        (
            allocation[
                "Allocated_Units"
            ]
            /
            allocation[
                "Net_Requirement"
            ]
        ),
        1.0,
    )

    allocation[
        "Item_Coverage_Pct"
    ] = (
        allocation[
            "Item_Coverage_Ratio"
        ]
        *
        100
    )

    allocation[
        "Required_Value"
    ] = (
        allocation[
            "Net_Requirement"
        ]
        *
        allocation[
            "Planning_Cost"
        ]
    )

    allocation[
        "Allocated_Value"
    ] = (
        allocation[
            "Allocated_Units"
        ]
        *
        allocation[
            "Planning_Cost"
        ]
    )

    allocation[
        "Unfunded_Requirement_Value"
    ] = (
        allocation[
            "Required_Value"
        ]
        -
        allocation[
            "Allocated_Value"
        ]
    )

    allocation[
        "Allocation_Status"
    ] = np.select(
        [
            allocation[
                "Net_Requirement"
            ]
            .eq(0),

            (
                allocation[
                    "Net_Requirement"
                ]
                .gt(0)
                &
                allocation[
                    "Allocated_Units"
                ]
                .eq(
                    allocation[
                        "Net_Requirement"
                    ]
                )
            ),

            (
                allocation[
                    "Allocated_Units"
                ]
                .gt(0)
            ),
        ],
        [
            "NO_NET_REQUIREMENT",
            "FULLY_FUNDED_MODELED_REQUIREMENT",
            "PARTIALLY_FUNDED_MODELED_REQUIREMENT",
        ],
        default=(
            "ZERO_ALLOCATION_POSITIVE_REQUIREMENT"
        ),
    )

    allocation[
        "Optimization_Version"
    ] = OPTIMIZATION_VERSION

    final_value = float(
        allocation[
            "Allocated_Value"
        ]
        .sum()
    )

    final_theta_a = float(
        value(
            model.theta_A
        )
    )

    final_theta_b = float(
        value(
            model.theta_B
        )
    )

    final_theta_c = float(
        value(
            model.theta_C
        )
    )

    # --------------------------------------------------------
    # OBSERVED MINIMUM COVERAGE
    # --------------------------------------------------------

    observed_min = {}

    for abc_class in ABC_CLASSES:

        class_positive = (
            allocation.loc[
                allocation[
                    "ABC_AsOf"
                ]
                .eq(
                    abc_class
                )
                &
                allocation[
                    "Net_Requirement"
                ]
                .gt(0)
            ]
        )

        if class_positive.empty:

            observed_min[
                abc_class
            ] = 1.0

        else:

            observed_min[
                abc_class
            ] = float(
                class_positive[
                    "Item_Coverage_Ratio"
                ]
                .min()
            )

    required_units = int(
        allocation[
            "Net_Requirement"
        ]
        .sum()
    )

    allocated_units = int(
        allocation[
            "Allocated_Units"
        ]
        .sum()
    )

    required_value = float(
        allocation[
            "Required_Value"
        ]
        .sum()
    )

    unfunded_units = int(
        allocation[
            "Unfunded_Requirement_Units"
        ]
        .sum()
    )

    unfunded_value = float(
        allocation[
            "Unfunded_Requirement_Value"
        ]
        .sum()
    )

    budget_utilization = (
        final_value
        /
        budget
        *
        100
        if budget > 0
        else 100.0
    )

    summary = {
        "Requirement_Scenario":
            scenario,

        "Budget_Factor":
            budget_factor,

        "Budget_Percent":
            budget_factor
            *
            100,

        "Eligible_Medicine_Count":
            len(
                allocation
            ),

        "Positive_Net_Requirement_Count":
            int(
                allocation[
                    "Net_Requirement"
                ]
                .gt(0)
                .sum()
            ),

        "Required_Units":
            required_units,

        "Allocated_Units":
            allocated_units,

        "Unfunded_Requirement_Units":
            unfunded_units,

        "B100":
            b100,

        "Scenario_Budget":
            budget,

        "Required_Value":
            required_value,

        "Allocated_Value":
            final_value,

        "Unfunded_Requirement_Value":
            unfunded_value,

        "Unspent_Budget":
            budget
            -
            final_value,

        "Budget_Utilization_Pct":
            budget_utilization,

        "Stage1_Theta_A":
            stage1_theta_a,

        "Stage2_Theta_B":
            stage2_theta_b,

        "Stage3_Theta_C":
            stage3_theta_c,

        "Final_Theta_A":
            final_theta_a,

        "Final_Theta_B":
            final_theta_b,

        "Final_Theta_C":
            final_theta_c,

        "Observed_Min_Coverage_A":
            observed_min["A"],

        "Observed_Min_Coverage_B":
            observed_min["B"],

        "Observed_Min_Coverage_C":
            observed_min["C"],

        "Stage4_Max_Allocated_Value":
            stage4_value,

        "Stage5_Final_Allocated_Value":
            final_value,

        "Stage5_Value_Difference":
            (
                final_value
                -
                stage4_value
            ),

        "Full_Funding_Row_Count":
            int(
                allocation[
                    "Allocation_Status"
                ]
                .eq(
                    "FULLY_FUNDED_MODELED_REQUIREMENT"
                )
                .sum()
            ),

        "Partial_Funding_Row_Count":
            int(
                allocation[
                    "Allocation_Status"
                ]
                .eq(
                    "PARTIALLY_FUNDED_MODELED_REQUIREMENT"
                )
                .sum()
            ),

        "Zero_Allocation_Positive_Row_Count":
            int(
                allocation[
                    "Allocation_Status"
                ]
                .eq(
                    "ZERO_ALLOCATION_POSITIVE_REQUIREMENT"
                )
                .sum()
            ),

        "No_Net_Requirement_Row_Count":
            int(
                allocation[
                    "Allocation_Status"
                ]
                .eq(
                    "NO_NET_REQUIREMENT"
                )
                .sum()
            ),

        "Optimization_Version":
            OPTIMIZATION_VERSION,
    }

    return (
        allocation,
        summary,
    )


# ============================================================
# CASE VALIDATION
# ============================================================

def validate_case(
    allocation: pd.DataFrame,
    summary: dict,
) -> None:

    if (
        allocation[
            "Allocated_Units"
        ]
        .lt(0)
        .any()
    ):

        raise RuntimeError(
            "Negative allocation detected."
        )

    if (
        allocation[
            "Allocated_Units"
        ]
        .gt(
            allocation[
                "Net_Requirement"
            ]
        )
        .any()
    ):

        raise RuntimeError(
            "Allocation exceeds "
            "net requirement."
        )

    if (
        allocation[
            "Unfunded_Requirement_Units"
        ]
        .lt(0)
        .any()
    ):

        raise RuntimeError(
            "Negative unfunded "
            "requirement detected."
        )

    if (
        summary[
            "Allocated_Value"
        ]
        >
        summary[
            "Scenario_Budget"
        ]
        +
        BUDGET_TOLERANCE
    ):

        raise RuntimeError(
            "Budget constraint violated."
        )

    if (
        summary[
            "Observed_Min_Coverage_A"
        ]
        +
        COVERAGE_TOLERANCE
        <
        summary[
            "Stage1_Theta_A"
        ]
    ):

        raise RuntimeError(
            "Category A coverage "
            "preservation failed."
        )

    if (
        summary[
            "Observed_Min_Coverage_B"
        ]
        +
        COVERAGE_TOLERANCE
        <
        summary[
            "Stage2_Theta_B"
        ]
    ):

        raise RuntimeError(
            "Category B coverage "
            "preservation failed."
        )

    if (
        summary[
            "Observed_Min_Coverage_C"
        ]
        +
        COVERAGE_TOLERANCE
        <
        summary[
            "Stage3_Theta_C"
        ]
    ):

        raise RuntimeError(
            "Category C coverage "
            "preservation failed."
        )

    if (
        summary[
            "Stage5_Final_Allocated_Value"
        ]
        <
        summary[
            "Stage4_Max_Allocated_Value"
        ]
        -
        VALUE_PRESERVE_TOLERANCE
        -
        1e-6
    ):

        raise RuntimeError(
            "Stage 5 sacrificed more "
            "allocated value than allowed."
        )


# ============================================================
# BY ABC SUMMARY
# ============================================================

def build_by_abc(
    allocations: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        scenario,
        budget_factor,
        abc_class,
    ), group in allocations.groupby(
        [
            "Requirement_Scenario",
            "Budget_Factor",
            "ABC_AsOf",
        ],
        sort=True,
    ):

        required_units = int(
            group[
                "Net_Requirement"
            ]
            .sum()
        )

        allocated_units = int(
            group[
                "Allocated_Units"
            ]
            .sum()
        )

        required_value = float(
            group[
                "Required_Value"
            ]
            .sum()
        )

        allocated_value = float(
            group[
                "Allocated_Value"
            ]
            .sum()
        )

        positive = (
            group.loc[
                group[
                    "Net_Requirement"
                ]
                .gt(0)
            ]
        )

        if positive.empty:

            minimum_coverage = 1.0

        else:

            minimum_coverage = float(
                positive[
                    "Item_Coverage_Ratio"
                ]
                .min()
            )

        rows.append(
            {
                "Requirement_Scenario":
                    scenario,

                "Budget_Factor":
                    budget_factor,

                "Budget_Percent":
                    budget_factor
                    *
                    100,

                "ABC_AsOf":
                    abc_class,

                "Medicine_Count":
                    len(
                        group
                    ),

                "Positive_Net_Count":
                    int(
                        group[
                            "Net_Requirement"
                        ]
                        .gt(0)
                        .sum()
                    ),

                "Required_Units":
                    required_units,

                "Allocated_Units":
                    allocated_units,

                "Unfunded_Requirement_Units":
                    (
                        required_units
                        -
                        allocated_units
                    ),

                "Unit_Coverage_Pct":
                    (
                        allocated_units
                        /
                        required_units
                        *
                        100
                        if required_units
                        >
                        0
                        else 100.0
                    ),

                "Minimum_Item_Coverage_Pct":
                    minimum_coverage
                    *
                    100,

                "Required_Value":
                    required_value,

                "Allocated_Value":
                    allocated_value,

                "Unfunded_Requirement_Value":
                    (
                        required_value
                        -
                        allocated_value
                    ),

                "Value_Coverage_Pct":
                    (
                        allocated_value
                        /
                        required_value
                        *
                        100
                        if required_value
                        >
                        0
                        else 100.0
                    ),

                "Fully_Funded_Count":
                    int(
                        group[
                            "Allocation_Status"
                        ]
                        .eq(
                            "FULLY_FUNDED_MODELED_REQUIREMENT"
                        )
                        .sum()
                    ),

                "Partially_Funded_Count":
                    int(
                        group[
                            "Allocation_Status"
                        ]
                        .eq(
                            "PARTIALLY_FUNDED_MODELED_REQUIREMENT"
                        )
                        .sum()
                    ),

                "Zero_Allocation_Positive_Count":
                    int(
                        group[
                            "Allocation_Status"
                        ]
                        .eq(
                            "ZERO_ALLOCATION_POSITIVE_REQUIREMENT"
                        )
                        .sum()
                    ),

                "Optimization_Version":
                    OPTIMIZATION_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MONOTONICITY VALIDATION
# ============================================================

def validate_budget_monotonicity(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    coverage_columns = [
        "Observed_Min_Coverage_A",
        "Observed_Min_Coverage_B",
        "Observed_Min_Coverage_C",
    ]

    for scenario, group in (
        summary.groupby(
            "Requirement_Scenario",
            sort=True,
        )
    ):

        ordered = (
            group.sort_values(
                "Budget_Factor"
            )
            .reset_index(
                drop=True
            )
        )

        for column in coverage_columns:

            values = (
                ordered[
                    column
                ]
                .to_numpy(
                    dtype=float
                )
            )

            differences = np.diff(
                values
            )

            passed = bool(
                (
                    differences
                    >=
                    -COVERAGE_TOLERANCE
                )
                .all()
            )

            rows.append(
                {
                    "Requirement_Scenario":
                        scenario,

                    "Metric":
                        column,

                    "Budget_Sequence":
                        "60%->70%->80%->90%",

                    "Values":
                        " | ".join(
                            f"{x:.8f}"
                            for x in values
                        ),

                    "Nondecreasing_PASS":
                        passed,

                    "Optimization_Version":
                        OPTIMIZATION_VERSION,
                }
            )

    result = pd.DataFrame(
        rows
    )

    if not (
        result[
            "Nondecreasing_PASS"
        ]
        .all()
    ):

        raise RuntimeError(
            "Coverage monotonicity "
            "validation FAILED."
        )

    return result


# ============================================================
# REPEATABILITY
# ============================================================

def allocation_signature(
    dataframe: pd.DataFrame,
) -> str:

    ordered = (
        dataframe[
            [
                "Medicine_Key",
                "Allocated_Units",
            ]
        ]
        .sort_values(
            "Medicine_Key"
        )
    )

    text = "\n".join(
        f"{row.Medicine_Key}:{int(row.Allocated_Units)}"
        for row in ordered.itertuples(
            index=False
        )
    )

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def build_repeatability_check(
    source: pd.DataFrame,
    first_solution: pd.DataFrame,
) -> pd.DataFrame:

    scenario = (
        REPEATABILITY_SCENARIO
    )

    budget_factor = (
        REPEATABILITY_BUDGET_FACTOR
    )

    requirement_column = (
        SCENARIOS[
            scenario
        ]
    )

    second_solution, second_summary = (
        solve_budget_case(
            dataframe=source,
            scenario=scenario,
            requirement_column=(
                requirement_column
            ),
            budget_factor=(
                budget_factor
            ),
        )
    )

    validate_case(
        second_solution,
        second_summary,
    )

    first = (
        first_solution[
            [
                "Medicine_Key",
                "Allocated_Units",
            ]
        ]
        .sort_values(
            "Medicine_Key"
        )
        .reset_index(
            drop=True
        )
    )

    second = (
        second_solution[
            [
                "Medicine_Key",
                "Allocated_Units",
            ]
        ]
        .sort_values(
            "Medicine_Key"
        )
        .reset_index(
            drop=True
        )
    )

    mismatch_count = int(
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

    first_signature = (
        allocation_signature(
            first_solution
        )
    )

    second_signature = (
        allocation_signature(
            second_solution
        )
    )

    passed = bool(
        mismatch_count
        ==
        0
        and
        first_signature
        ==
        second_signature
    )

    if not passed:

        raise RuntimeError(
            "Repeatability check FAILED."
        )

    return pd.DataFrame(
        [
            {
                "Requirement_Scenario":
                    scenario,

                "Budget_Factor":
                    budget_factor,

                "Budget_Percent":
                    budget_factor
                    *
                    100,

                "Repeated_Solve_Mismatch_Count":
                    mismatch_count,

                "First_Allocation_SHA256":
                    first_signature,

                "Second_Allocation_SHA256":
                    second_signature,

                "Exact_Repeatability_PASS":
                    passed,

                "Tie_Break_Rule":
                    (
                        "Stable Medicine_Key rank "
                        "after preserving maximum "
                        "allocated value."
                    ),

                "Tie_Break_Has_Clinical_Meaning":
                    False,

                "Tie_Break_Has_Economic_Priority_Meaning":
                    False,

                "Optimization_Version":
                    OPTIMIZATION_VERSION,
            }
        ]
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance(
    repeatability: pd.DataFrame,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "Optimization_Version":
                    OPTIMIZATION_VERSION,

                "Target_Year":
                    2026,

                "Pyomo_Version":
                    pyomo.__version__,

                "HiGHS_Version":
                    (
                        highspy.Highs()
                        .version()
                    ),

                "Solver_Interface":
                    "appsi_highs",

                "Optimization_Scope":
                    "ELIGIBLE_ABC_RECORDS_ONLY",

                "Eligible_Medicine_Count":
                    EXPECTED_ELIGIBLE,

                "Budget_Levels_Run":
                    "60%,70%,80%,90%",

                "Budget_Levels_Are_Actual_CDH_Budgets":
                    False,

                "Stage_1":
                    "MAX_MIN_A_COVERAGE",

                "Stage_2":
                    "PRESERVE_A_MAX_MIN_B",

                "Stage_3":
                    "PRESERVE_A_B_MAX_MIN_C",

                "Stage_4":
                    "MAXIMIZE_ALLOCATED_VALUE",

                "Stage_5":
                    (
                        "TECHNICAL_STABLE_"
                        "MEDICINE_KEY_TIE_BREAK"
                    ),

                "Technical_Tie_Break_Is_Clinical_Priority":
                    False,

                "Technical_Tie_Break_Is_Economic_Priority":
                    False,

                "Repeatability_Test_Case":
                    "POINT_80_PERCENT",

                "Repeatability_PASS":
                    bool(
                        repeatability[
                            "Exact_Repeatability_PASS"
                        ]
                        .iloc[0]
                    ),

                "Full_Hospital_Optimization_Claim":
                    False,

                "Reason_Full_Claim_Blocked":
                    (
                        "Five A/B medicines retain "
                        "unresolved baseline stock."
                    ),

                "Automatic_Purchase_Order":
                    False,
            }
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    allocations: pd.DataFrame,
    summary: pd.DataFrame,
    by_abc: pd.DataFrame,
    monotonicity: pd.DataFrame,
    repeatability: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "allocations_df",
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
            "monotonicity_df",
            monotonicity,
        )

        connection.register(
            "repeatability_df",
            repeatability,
        )

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_budget_allocations AS
            SELECT *
            FROM allocations_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_budget_summary AS
            SELECT *
            FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_budget_by_abc AS
            SELECT *
            FROM abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_budget_monotonicity AS
            SELECT *
            FROM monotonicity_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_repeatability AS
            SELECT *
            FROM repeatability_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            planning_2026_milp_budget_governance AS
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

    print("=" * 116)
    print("CDH INVENTORY DSS")
    print("2026 BUDGET-CONSTRAINED LEXICOGRAPHIC MILP")
    print("=" * 116)
    print()

    source = (
        load_inputs()
    )

    validate_source(
        source
    )

    print(
        "MILP source validation: PASS"
    )

    print(
        "Eligible A/B/C medicines:",
        f"{len(source):,}",
    )

    print(
        "Budget levels:",
        "60%, 70%, 80%, 90%",
    )

    print(
        "Requirement policies:",
        ", ".join(
            SCENARIOS.keys()
        ),
    )

    print(
        "Total constrained scenarios:",
        len(
            SCENARIOS
        )
        *
        len(
            BUDGET_FACTORS
        ),
    )

    print()

    allocation_frames = []
    summary_rows = []

    for scenario, requirement_column in (
        SCENARIOS.items()
    ):

        for budget_factor in (
            BUDGET_FACTORS
        ):

            print("-" * 116)

            print(
                "Solving:",
                scenario,
                f"@ {budget_factor:.0%} B100",
            )

            allocation, summary = (
                solve_budget_case(
                    dataframe=source,
                    scenario=scenario,
                    requirement_column=(
                        requirement_column
                    ),
                    budget_factor=(
                        budget_factor
                    ),
                )
            )

            validate_case(
                allocation,
                summary,
            )

            allocation_frames.append(
                allocation
            )

            summary_rows.append(
                summary
            )

            print(
                "  A minimum coverage:",
                f"{summary['Observed_Min_Coverage_A'] * 100:,.4f}%",
            )

            print(
                "  B minimum coverage:",
                f"{summary['Observed_Min_Coverage_B'] * 100:,.4f}%",
            )

            print(
                "  C minimum coverage:",
                f"{summary['Observed_Min_Coverage_C'] * 100:,.4f}%",
            )

            print(
                "  Budget:",
                f"{summary['Scenario_Budget']:,.2f}",
            )

            print(
                "  Allocated value:",
                f"{summary['Allocated_Value']:,.2f}",
            )

            print(
                "  Budget utilization:",
                f"{summary['Budget_Utilization_Pct']:,.4f}%",
            )

            print(
                "  Unfunded modeled "
                "requirement units:",
                f"{summary['Unfunded_Requirement_Units']:,}",
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

    monotonicity = (
        validate_budget_monotonicity(
            summary
        )
    )

    print()
    print(
        "Budget-to-coverage "
        "monotonicity validation: PASS"
    )

    first_repeatability_solution = (
        allocations.loc[
            allocations[
                "Requirement_Scenario"
            ]
            .eq(
                REPEATABILITY_SCENARIO
            )
            &
            np.isclose(
                allocations[
                    "Budget_Factor"
                ],
                REPEATABILITY_BUDGET_FACTOR,
            )
        ]
        .copy()
    )

    print(
        "Running independent "
        "POINT @ 80% repeatability check..."
    )

    repeatability = (
        build_repeatability_check(
            source=source,
            first_solution=(
                first_repeatability_solution
            ),
        )
    )

    print(
        "Exact allocation repeatability: PASS"
    )

    governance = (
        build_governance(
            repeatability
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    allocations.to_parquet(
        OUTPUT_DIR
        / "milp_budget_allocations.parquet",
        index=False,
        engine="pyarrow",
    )

    allocations.to_csv(
        OUTPUT_DIR
        / "milp_budget_allocations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "milp_budget_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "milp_budget_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    monotonicity.to_csv(
        OUTPUT_DIR
        / "milp_budget_monotonicity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeatability.to_csv(
        OUTPUT_DIR
        / "milp_repeatability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "milp_budget_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        allocations=allocations,
        summary=summary,
        by_abc=by_abc,
        monotonicity=monotonicity,
        repeatability=repeatability,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 116)
    print("BUDGET-CONSTRAINED OPTIMIZATION SUMMARY")
    print("=" * 116)

    display_columns = [
        "Requirement_Scenario",
        "Budget_Percent",
        "Required_Units",
        "Allocated_Units",
        "Unfunded_Requirement_Units",
        "Scenario_Budget",
        "Allocated_Value",
        "Unspent_Budget",
        "Budget_Utilization_Pct",
        "Observed_Min_Coverage_A",
        "Observed_Min_Coverage_B",
        "Observed_Min_Coverage_C",
        "Full_Funding_Row_Count",
        "Partial_Funding_Row_Count",
        "Zero_Allocation_Positive_Row_Count",
    ]

    print(
        summary[
            display_columns
        ]
        .sort_values(
            [
                "Requirement_Scenario",
                "Budget_Percent",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.6f}"
            ),
        )
    )

    print()
    print("=" * 116)
    print("ABC COVERAGE SUMMARY")
    print("=" * 116)

    print(
        by_abc[
            [
                "Requirement_Scenario",
                "Budget_Percent",
                "ABC_AsOf",
                "Positive_Net_Count",
                "Required_Units",
                "Allocated_Units",
                "Unit_Coverage_Pct",
                "Minimum_Item_Coverage_Pct",
                "Required_Value",
                "Allocated_Value",
                "Value_Coverage_Pct",
                "Fully_Funded_Count",
                "Partially_Funded_Count",
                "Zero_Allocation_Positive_Count",
            ]
        ]
        .sort_values(
            [
                "Requirement_Scenario",
                "Budget_Percent",
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
    print("=" * 116)
    print("REPEATABILITY CHECK")
    print("=" * 116)

    print(
        repeatability.to_string(
            index=False
        )
    )

    print()
    print("=" * 116)
    print("OPTIMIZATION GOVERNANCE")
    print("=" * 116)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 116)
    print("INTERPRETATION")
    print("-" * 116)

    print(
        "Coverage refers to fulfillment "
        "of modeled net procurement "
        "requirements, not clinical "
        "service-level coverage."
    )

    print(
        "Unfunded requirement is a "
        "modeled budget-allocation result; "
        "it is not an observed stockout."
    )

    print(
        "A>B>C priority is economic ABC "
        "priority only and must not be "
        "interpreted as clinical criticality."
    )

    print(
        "The final technical tie-break "
        "uses Medicine_Key ordering only "
        "after A/B/C coverage and maximum "
        "allocated value are preserved."
    )

    print(
        "The tie-break carries no clinical "
        "or economic-priority meaning."
    )

    print(
        "All budgets are modeled scenario "
        "budgets rather than actual "
        "historical CDH budgets."
    )

    print(
        "Results cover eligible A/B/C "
        "records only because five A/B "
        "medicines still have unresolved "
        "baseline stock."
    )

    print(
        "No purchase order is generated."
    )

    print()
    print("=" * 116)
    print(
        "BUDGET-CONSTRAINED MILP COMPLETE"
    )
    print("=" * 116)


if __name__ == "__main__":
    main()