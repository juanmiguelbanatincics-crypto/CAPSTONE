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

try:
    from .runtime_context import (
        load_runtime_context_from_env,
    )
except ImportError:
    from runtime_context import (
        load_runtime_context_from_env,
    )


# ============================================================
# RUNTIME PATHS
# ============================================================

RUNTIME_CONTEXT = (
    load_runtime_context_from_env()
)

DATABASE_PATH = (
    RUNTIME_CONTEXT.database_path
)

TARGET_YEAR = int(
    RUNTIME_CONTEXT.planning_year
)

SOURCE_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_inputs"
)

SOURCE_OPTIMIZER_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_optimizer_inputs"
)

OUTPUT_DIR = (
    RUNTIME_CONTEXT.processed_dir
    / "planning"
    / (
        f"milp_strict_abc_{TARGET_YEAR}_"
        "governed_v2"
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOCATIONS_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_strict_allocations"
)

SUMMARY_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_strict_summary"
)

BY_ABC_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_strict_by_abc"
)

PRIORITY_AUDIT_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_strict_priority_audit"
)

MONOTONICITY_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_strict_monotonicity"
)

REPEATABILITY_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_strict_repeatability"
)

GOVERNANCE_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_strict_governance"
)


# ============================================================
# GOVERNANCE
# ============================================================

OPTIMIZATION_VERSION = (
    f"MILP_STRICT_ABC_OPTIMIZATION_"
    f"{TARGET_YEAR}_GOVERNED_V2"
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

THETA_TOLERANCE = 1e-7
VALUE_TOLERANCE = 0.01
BUDGET_TOLERANCE = 0.05
COVERAGE_TOLERANCE = 1e-5

REPEATABILITY_SCENARIO = "POINT"
REPEATABILITY_FACTOR = 0.80


# ============================================================
# LOAD
# ============================================================

def load_inputs() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            f"""
            SELECT *
            FROM {SOURCE_TABLE}
            WHERE Optimization_Priority_Eligible = TRUE
            ORDER BY ABC_Priority_Order, Medicine_Key
            """
        ).df()

    finally:
        connection.close()

    return dataframe


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    dataframe: pd.DataFrame,
) -> None:

    if dataframe.empty:
        raise ValueError(
            "No strict A/B/C eligible rows "
            "were available for optimization."
        )

    if dataframe[
        "Medicine_Key"
    ].duplicated().any():

        raise ValueError(
            "Duplicate Medicine_Key."
        )

    if not dataframe[
        "ABC_AsOf"
    ].isin(
        ABC_CLASSES
    ).all():

        raise ValueError(
            "Non-A/B/C row detected."
        )

    if dataframe[
        "Planning_Cost"
    ].isna().any():

        raise ValueError(
            "Missing planning cost."
        )

    if dataframe[
        "Planning_Cost"
    ].le(0).any():

        raise ValueError(
            "Non-positive planning cost."
        )

    for column in SCENARIOS.values():

        if dataframe[
            column
        ].isna().any():

            raise ValueError(
                f"Missing {column}."
            )

        values = dataframe[
            column
        ].astype(float).to_numpy()

        if (values < 0).any():

            raise ValueError(
                f"Negative {column}."
            )

        if not np.allclose(
            values,
            np.round(values),
            atol=1e-9,
            rtol=0,
        ):

            raise ValueError(
                f"{column} is not integer."
            )


# ============================================================
# MODEL
# ============================================================

def build_model(
    dataframe: pd.DataFrame,
    requirement_column: str,
    budget_factor: float,
):

    keys = dataframe[
        "Medicine_Key"
    ].tolist()

    costs = dict(
        zip(
            keys,
            dataframe[
                "Planning_Cost"
            ].astype(float),
        )
    )

    requirements = dict(
        zip(
            keys,
            dataframe[
                requirement_column
            ].astype(int),
        )
    )

    abc = dict(
        zip(
            keys,
            dataframe[
                "ABC_AsOf"
            ],
        )
    )

    sorted_keys = sorted(keys)

    tie_weights = {
        key:
            len(sorted_keys) - rank
        for rank, key
        in enumerate(sorted_keys)
    }

    b100 = float(
        sum(
            costs[key]
            *
            requirements[key]
            for key in keys
        )
    )

    budget = float(
        b100
        *
        budget_factor
    )

    positive_by_class = {
        abc_class: [
            key
            for key in keys
            if (
                abc[key] == abc_class
                and
                requirements[key] > 0
            )
        ]
        for abc_class
        in ABC_CLASSES
    }

    class_all = {
        abc_class: [
            key
            for key in keys
            if abc[key] == abc_class
        ]
        for abc_class
        in ABC_CLASSES
    }

    m = ConcreteModel()

    m.I = Set(
        initialize=keys,
        ordered=True,
    )

    m.R = Param(
        m.I,
        initialize=requirements,
        within=NonNegativeIntegers,
    )

    m.C = Param(
        m.I,
        initialize=costs,
    )

    m.W = Param(
        m.I,
        initialize=tie_weights,
        within=NonNegativeIntegers,
    )

    m.x = Var(
        m.I,
        domain=NonNegativeIntegers,
        bounds=lambda model, i: (
            0,
            int(
                value(
                    model.R[i]
                )
            ),
        ),
    )

    # --------------------------------------------------------
    # BUDGET
    # --------------------------------------------------------

    m.total_value = Expression(
        expr=sum(
            m.C[i]
            *
            m.x[i]
            for i in m.I
        )
    )

    m.budget = Constraint(
        expr=(
            m.total_value
            <=
            budget
        )
    )

    # --------------------------------------------------------
    # ABC SETS
    # --------------------------------------------------------

    m.A_POS = Set(
        initialize=(
            positive_by_class["A"]
        )
    )

    m.B_POS = Set(
        initialize=(
            positive_by_class["B"]
        )
    )

    m.C_POS = Set(
        initialize=(
            positive_by_class["C"]
        )
    )

    m.A_ALL = Set(
        initialize=(
            class_all["A"]
        )
    )

    m.B_ALL = Set(
        initialize=(
            class_all["B"]
        )
    )

    m.C_ALL = Set(
        initialize=(
            class_all["C"]
        )
    )

    # --------------------------------------------------------
    # COVERAGE VARIABLES
    # --------------------------------------------------------

    m.theta_A = Var(
        domain=UnitInterval
    )

    m.theta_B = Var(
        domain=UnitInterval
    )

    m.theta_C = Var(
        domain=UnitInterval
    )

    if not positive_by_class["A"]:
        m.theta_A.fix(1.0)

    if not positive_by_class["B"]:
        m.theta_B.fix(1.0)

    if not positive_by_class["C"]:
        m.theta_C.fix(1.0)

    # --------------------------------------------------------
    # MINIMUM COVERAGE CONSTRAINTS
    # --------------------------------------------------------

    def coverage_a_rule(
        model,
        i,
    ):
        return (
            model.x[i]
            >=
            model.theta_A
            *
            model.R[i]
        )

    def coverage_b_rule(
        model,
        i,
    ):
        return (
            model.x[i]
            >=
            model.theta_B
            *
            model.R[i]
        )

    def coverage_c_rule(
        model,
        i,
    ):
        return (
            model.x[i]
            >=
            model.theta_C
            *
            model.R[i]
        )

    m.coverage_A = Constraint(
        m.A_POS,
        rule=coverage_a_rule,
    )

    m.coverage_B = Constraint(
        m.B_POS,
        rule=coverage_b_rule,
    )

    m.coverage_C = Constraint(
        m.C_POS,
        rule=coverage_c_rule,
    )

    # --------------------------------------------------------
    # CLASS VALUE EXPRESSIONS
    # --------------------------------------------------------

    m.value_A = Expression(
        expr=sum(
            m.C[i] * m.x[i]
            for i in m.A_ALL
        )
    )

    m.value_B = Expression(
        expr=sum(
            m.C[i] * m.x[i]
            for i in m.B_ALL
        )
    )

    m.value_C = Expression(
        expr=sum(
            m.C[i] * m.x[i]
            for i in m.C_ALL
        )
    )

    # --------------------------------------------------------
    # TECHNICAL TIE SCORE
    # --------------------------------------------------------

    m.tie_score = Expression(
        expr=sum(
            m.W[i] * m.x[i]
            for i in m.I
        )
    )

    # --------------------------------------------------------
    # SEVEN OBJECTIVES
    # --------------------------------------------------------

    m.obj_theta_A = Objective(
        expr=m.theta_A,
        sense=maximize,
    )

    m.obj_value_A = Objective(
        expr=m.value_A,
        sense=maximize,
    )

    m.obj_theta_B = Objective(
        expr=m.theta_B,
        sense=maximize,
    )

    m.obj_value_B = Objective(
        expr=m.value_B,
        sense=maximize,
    )

    m.obj_theta_C = Objective(
        expr=m.theta_C,
        sense=maximize,
    )

    m.obj_value_C = Objective(
        expr=m.value_C,
        sense=maximize,
    )

    m.obj_tie = Objective(
        expr=m.tie_score,
        sense=maximize,
    )

    m.obj_value_A.deactivate()
    m.obj_theta_B.deactivate()
    m.obj_value_B.deactivate()
    m.obj_theta_C.deactivate()
    m.obj_value_C.deactivate()
    m.obj_tie.deactivate()

    return (
        m,
        b100,
        budget,
    )


# ============================================================
# SOLVER
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


def solve_optimal(
    solver,
    model,
    name: str,
):

    result = solver.solve(
        model
    )

    termination = (
        result.solver
        .termination_condition
    )

    if (
        termination
        !=
        TerminationCondition.optimal
    ):

        raise RuntimeError(
            f"{name}: solver termination "
            f"{termination}"
        )


# ============================================================
# ONE STRICT LEXICOGRAPHIC CASE
# ============================================================

def solve_case(
    dataframe: pd.DataFrame,
    scenario: str,
    requirement_column: str,
    budget_factor: float,
):

    m, b100, budget = (
        build_model(
            dataframe,
            requirement_column,
            budget_factor,
        )
    )

    solver = make_solver()

    # ========================================================
    # 1 — MAX MINIMUM A COVERAGE
    # ========================================================

    solve_optimal(
        solver,
        m,
        "Stage 1 theta A",
    )

    theta_a_star = float(
        value(
            m.theta_A
        )
    )

    m.keep_theta_A = Constraint(
        expr=(
            m.theta_A
            >=
            max(
                0.0,
                theta_a_star
                -
                THETA_TOLERANCE,
            )
        )
    )

    m.obj_theta_A.deactivate()
    m.obj_value_A.activate()

    # ========================================================
    # 2 — MAXIMIZE A VALUE
    # ========================================================

    solve_optimal(
        solver,
        m,
        "Stage 2 A value",
    )

    value_a_star = float(
        value(
            m.value_A
        )
    )

    m.keep_value_A = Constraint(
        expr=(
            m.value_A
            >=
            value_a_star
            -
            VALUE_TOLERANCE
        )
    )

    m.obj_value_A.deactivate()
    m.obj_theta_B.activate()

    # ========================================================
    # 3 — MAX MINIMUM B COVERAGE
    # ========================================================

    solve_optimal(
        solver,
        m,
        "Stage 3 theta B",
    )

    theta_b_star = float(
        value(
            m.theta_B
        )
    )

    m.keep_theta_B = Constraint(
        expr=(
            m.theta_B
            >=
            max(
                0.0,
                theta_b_star
                -
                THETA_TOLERANCE,
            )
        )
    )

    m.obj_theta_B.deactivate()
    m.obj_value_B.activate()

    # ========================================================
    # 4 — MAXIMIZE B VALUE
    # ========================================================

    solve_optimal(
        solver,
        m,
        "Stage 4 B value",
    )

    value_b_star = float(
        value(
            m.value_B
        )
    )

    m.keep_value_B = Constraint(
        expr=(
            m.value_B
            >=
            value_b_star
            -
            VALUE_TOLERANCE
        )
    )

    m.obj_value_B.deactivate()
    m.obj_theta_C.activate()

    # ========================================================
    # 5 — MAX MINIMUM C COVERAGE
    # ========================================================

    solve_optimal(
        solver,
        m,
        "Stage 5 theta C",
    )

    theta_c_star = float(
        value(
            m.theta_C
        )
    )

    m.keep_theta_C = Constraint(
        expr=(
            m.theta_C
            >=
            max(
                0.0,
                theta_c_star
                -
                THETA_TOLERANCE,
            )
        )
    )

    m.obj_theta_C.deactivate()
    m.obj_value_C.activate()

    # ========================================================
    # 6 — MAXIMIZE C VALUE
    # ========================================================

    solve_optimal(
        solver,
        m,
        "Stage 6 C value",
    )

    value_c_star = float(
        value(
            m.value_C
        )
    )

    m.keep_value_C = Constraint(
        expr=(
            m.value_C
            >=
            value_c_star
            -
            VALUE_TOLERANCE
        )
    )

    m.obj_value_C.deactivate()
    m.obj_tie.activate()

    # ========================================================
    # 7 — DETERMINISTIC TIE BREAK
    # ========================================================

    solve_optimal(
        solver,
        m,
        "Stage 7 tie break",
    )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    result = dataframe[
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

    result[
        "Requirement_Scenario"
    ] = scenario

    result[
        "Budget_Factor"
    ] = budget_factor

    result[
        "Budget_Percent"
    ] = (
        budget_factor * 100
    )

    result[
        "B100"
    ] = b100

    result[
        "Scenario_Budget"
    ] = budget

    result[
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

    result[
        "Allocated_Units"
    ] = [
        int(
            round(
                value(
                    m.x[key]
                )
            )
        )
        for key in dataframe[
            "Medicine_Key"
        ]
    ]

    result[
        "Unfunded_Requirement_Units"
    ] = (
        result[
            "Net_Requirement"
        ]
        -
        result[
            "Allocated_Units"
        ]
    )

    result[
        "Required_Value"
    ] = (
        result[
            "Net_Requirement"
        ]
        *
        result[
            "Planning_Cost"
        ]
    )

    result[
        "Allocated_Value"
    ] = (
        result[
            "Allocated_Units"
        ]
        *
        result[
            "Planning_Cost"
        ]
    )

    result[
        "Unfunded_Value"
    ] = (
        result[
            "Required_Value"
        ]
        -
        result[
            "Allocated_Value"
        ]
    )

    result[
        "Item_Coverage_Ratio"
    ] = np.where(
        result[
            "Net_Requirement"
        ].gt(0),
        (
            result[
                "Allocated_Units"
            ]
            /
            result[
                "Net_Requirement"
            ]
        ),
        1.0,
    )

    result[
        "Optimization_Version"
    ] = (
        OPTIMIZATION_VERSION
    )

    # --------------------------------------------------------
    # FINAL CLASS METRICS
    # --------------------------------------------------------

    class_metrics = {}

    for abc_class in ABC_CLASSES:

        group = result.loc[
            result[
                "ABC_AsOf"
            ].eq(
                abc_class
            )
        ]

        positive = group.loc[
            group[
                "Net_Requirement"
            ].gt(0)
        ]

        if positive.empty:
            min_coverage = 1.0
        else:
            min_coverage = float(
                positive[
                    "Item_Coverage_Ratio"
                ]
                .min()
            )

        class_metrics[
            abc_class
        ] = {
            "required_value":
                float(
                    group[
                        "Required_Value"
                    ].sum()
                ),

            "allocated_value":
                float(
                    group[
                        "Allocated_Value"
                    ].sum()
                ),

            "required_units":
                int(
                    group[
                        "Net_Requirement"
                    ].sum()
                ),

            "allocated_units":
                int(
                    group[
                        "Allocated_Units"
                    ].sum()
                ),

            "minimum_coverage":
                min_coverage,
        }

    total_allocated_value = float(
        result[
            "Allocated_Value"
        ].sum()
    )

    summary = {
        "Requirement_Scenario":
            scenario,

        "Budget_Factor":
            budget_factor,

        "Budget_Percent":
            budget_factor * 100,

        "B100":
            b100,

        "Scenario_Budget":
            budget,

        "Allocated_Value":
            total_allocated_value,

        "Unspent_Budget":
            (
                budget
                -
                total_allocated_value
            ),

        "Budget_Utilization_Pct":
            (
                total_allocated_value
                /
                budget
                *
                100
                if budget > 0
                else 100.0
            ),

        "Required_Units":
            int(
                result[
                    "Net_Requirement"
                ].sum()
            ),

        "Allocated_Units":
            int(
                result[
                    "Allocated_Units"
                ].sum()
            ),

        "Unfunded_Units":
            int(
                result[
                    "Unfunded_Requirement_Units"
                ].sum()
            ),

        "Theta_A_Star":
            theta_a_star,

        "Max_A_Value_Star":
            value_a_star,

        "Final_A_Value":
            class_metrics[
                "A"
            ][
                "allocated_value"
            ],

        "Observed_Min_A":
            class_metrics[
                "A"
            ][
                "minimum_coverage"
            ],

        "Theta_B_Star":
            theta_b_star,

        "Max_B_Value_Star":
            value_b_star,

        "Final_B_Value":
            class_metrics[
                "B"
            ][
                "allocated_value"
            ],

        "Observed_Min_B":
            class_metrics[
                "B"
            ][
                "minimum_coverage"
            ],

        "Theta_C_Star":
            theta_c_star,

        "Max_C_Value_Star":
            value_c_star,

        "Final_C_Value":
            class_metrics[
                "C"
            ][
                "allocated_value"
            ],

        "Observed_Min_C":
            class_metrics[
                "C"
            ][
                "minimum_coverage"
            ],

        "A_Value_Preservation_PASS":
            bool(
                class_metrics[
                    "A"
                ][
                    "allocated_value"
                ]
                >=
                value_a_star
                -
                VALUE_TOLERANCE
                -
                1e-6
            ),

        "B_Value_Preservation_PASS":
            bool(
                class_metrics[
                    "B"
                ][
                    "allocated_value"
                ]
                >=
                value_b_star
                -
                VALUE_TOLERANCE
                -
                1e-6
            ),

        "C_Value_Preservation_PASS":
            bool(
                class_metrics[
                    "C"
                ][
                    "allocated_value"
                ]
                >=
                value_c_star
                -
                VALUE_TOLERANCE
                -
                1e-6
            ),

        "Optimization_Version":
            OPTIMIZATION_VERSION,
    }

    return (
        result,
        summary,
    )


# ============================================================
# VALIDATE ONE CASE
# ============================================================

def validate_case(
    allocation: pd.DataFrame,
    summary: dict,
):

    if allocation[
        "Allocated_Units"
    ].lt(0).any():

        raise RuntimeError(
            "Negative allocation."
        )

    if allocation[
        "Allocated_Units"
    ].gt(
        allocation[
            "Net_Requirement"
        ]
    ).any():

        raise RuntimeError(
            "Allocation exceeds requirement."
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
            "Budget violation."
        )

    if (
        summary[
            "Observed_Min_A"
        ]
        +
        COVERAGE_TOLERANCE
        <
        summary[
            "Theta_A_Star"
        ]
    ):

        raise RuntimeError(
            "A minimum coverage not preserved."
        )

    if (
        summary[
            "Observed_Min_B"
        ]
        +
        COVERAGE_TOLERANCE
        <
        summary[
            "Theta_B_Star"
        ]
    ):

        raise RuntimeError(
            "B minimum coverage not preserved."
        )

    if (
        summary[
            "Observed_Min_C"
        ]
        +
        COVERAGE_TOLERANCE
        <
        summary[
            "Theta_C_Star"
        ]
    ):

        raise RuntimeError(
            "C minimum coverage not preserved."
        )

    for column in [
        "A_Value_Preservation_PASS",
        "B_Value_Preservation_PASS",
        "C_Value_Preservation_PASS",
    ]:

        if not summary[
            column
        ]:

            raise RuntimeError(
                f"{column} failed."
            )


# ============================================================
# BY ABC
# ============================================================

def build_by_abc(
    allocations: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        scenario,
        factor,
        abc_class,
    ), group in allocations.groupby(
        [
            "Requirement_Scenario",
            "Budget_Factor",
            "ABC_AsOf",
        ],
        sort=True,
    ):

        positive = group.loc[
            group[
                "Net_Requirement"
            ].gt(0)
        ]

        if positive.empty:
            min_coverage = 1.0
        else:
            min_coverage = float(
                positive[
                    "Item_Coverage_Ratio"
                ]
                .min()
            )

        required_units = int(
            group[
                "Net_Requirement"
            ].sum()
        )

        allocated_units = int(
            group[
                "Allocated_Units"
            ].sum()
        )

        required_value = float(
            group[
                "Required_Value"
            ].sum()
        )

        allocated_value = float(
            group[
                "Allocated_Value"
            ].sum()
        )

        rows.append(
            {
                "Requirement_Scenario":
                    scenario,

                "Budget_Factor":
                    factor,

                "Budget_Percent":
                    factor * 100,

                "ABC_AsOf":
                    abc_class,

                "Positive_Net_Count":
                    int(
                        positive.shape[0]
                    ),

                "Required_Units":
                    required_units,

                "Allocated_Units":
                    allocated_units,

                "Unfunded_Units":
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
                        if required_units > 0
                        else 100.0
                    ),

                "Minimum_Item_Coverage_Pct":
                    (
                        min_coverage
                        *
                        100
                    ),

                "Required_Value":
                    required_value,

                "Allocated_Value":
                    allocated_value,

                "Value_Coverage_Pct":
                    (
                        allocated_value
                        /
                        required_value
                        *
                        100
                        if required_value > 0
                        else 100.0
                    ),

                "Optimization_Version":
                    OPTIMIZATION_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRIORITY AUDIT
# ============================================================

def build_priority_audit(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in (
        summary.iterrows()
    ):

        rows.append(
            {
                "Requirement_Scenario":
                    row[
                        "Requirement_Scenario"
                    ],

                "Budget_Percent":
                    row[
                        "Budget_Percent"
                    ],

                "A_Min_Coverage_Preserved":
                    bool(
                        row[
                            "Observed_Min_A"
                        ]
                        +
                        COVERAGE_TOLERANCE
                        >=
                        row[
                            "Theta_A_Star"
                        ]
                    ),

                "A_Max_Value_Preserved":
                    bool(
                        row[
                            "A_Value_Preservation_PASS"
                        ]
                    ),

                "B_Min_Coverage_Preserved":
                    bool(
                        row[
                            "Observed_Min_B"
                        ]
                        +
                        COVERAGE_TOLERANCE
                        >=
                        row[
                            "Theta_B_Star"
                        ]
                    ),

                "B_Max_Value_Preserved":
                    bool(
                        row[
                            "B_Value_Preservation_PASS"
                        ]
                    ),

                "C_Min_Coverage_Preserved":
                    bool(
                        row[
                            "Observed_Min_C"
                        ]
                        +
                        COVERAGE_TOLERANCE
                        >=
                        row[
                            "Theta_C_Star"
                        ]
                    ),

                "C_Max_Value_Preserved":
                    bool(
                        row[
                            "C_Value_Preservation_PASS"
                        ]
                    ),

                "Strict_ABC_Priority_PASS":
                    bool(
                        row[
                            "A_Value_Preservation_PASS"
                        ]
                        and
                        row[
                            "B_Value_Preservation_PASS"
                        ]
                        and
                        row[
                            "C_Value_Preservation_PASS"
                        ]
                    ),

                "Optimization_Version":
                    OPTIMIZATION_VERSION,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result[
        "Strict_ABC_Priority_PASS"
    ].all():

        raise RuntimeError(
            "Strict ABC priority audit failed."
        )

    return result


# ============================================================
# BUDGET MONOTONICITY
# ============================================================

def validate_monotonicity(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for scenario, group in (
        summary.groupby(
            "Requirement_Scenario",
            sort=True,
        )
    ):

        ordered = group.sort_values(
            "Budget_Factor"
        )

        for column in [
            "Observed_Min_A",
            "Observed_Min_B",
            "Observed_Min_C",
        ]:

            values = ordered[
                column
            ].to_numpy(
                dtype=float
            )

            passed = bool(
                (
                    np.diff(values)
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

                    "Values":
                        " | ".join(
                            f"{x:.8f}"
                            for x
                            in values
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

    if not result[
        "Nondecreasing_PASS"
    ].all():

        raise RuntimeError(
            "Budget monotonicity failed."
        )

    return result


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
    source: pd.DataFrame,
    first: pd.DataFrame,
) -> pd.DataFrame:

    second, second_summary = (
        solve_case(
            dataframe=source,
            scenario=(
                REPEATABILITY_SCENARIO
            ),
            requirement_column=(
                SCENARIOS[
                    REPEATABILITY_SCENARIO
                ]
            ),
            budget_factor=(
                REPEATABILITY_FACTOR
            ),
        )
    )

    validate_case(
        second,
        second_summary,
    )

    first_sorted = first.sort_values(
        "Medicine_Key"
    )

    second_sorted = second.sort_values(
        "Medicine_Key"
    )

    mismatches = int(
        (
            first_sorted[
                "Allocated_Units"
            ]
            .to_numpy()
            !=
            second_sorted[
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
            "Strict V2 repeatability failed."
        )

    return pd.DataFrame(
        [
            {
                "Scenario":
                    REPEATABILITY_SCENARIO,

                "Budget_Percent":
                    REPEATABILITY_FACTOR
                    *
                    100,

                "Allocation_Mismatch_Count":
                    mismatches,

                "First_SHA256":
                    first_hash,

                "Second_SHA256":
                    second_hash,

                "Exact_Repeatability_PASS":
                    passed,

                "Optimization_Version":
                    OPTIMIZATION_VERSION,
            }
        ]
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance(
    source: pd.DataFrame,
) -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        unresolved_ab = int(
            connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {SOURCE_OPTIMIZER_TABLE}
                WHERE
                    High_Priority_AB_Unresolved = TRUE
                """
            ).fetchone()[0]
        )

    finally:
        connection.close()

    return pd.DataFrame(
        [
            {
                "Optimization_Version":
                    OPTIMIZATION_VERSION,

                "Target_Year":
                    TARGET_YEAR,

                "Eligible_ABC_Rows":
                    len(source),

                "Pyomo_Version":
                    pyomo.__version__,

                "HiGHS_Version":
                    highspy.Highs().version(),

                "Solver_Interface":
                    "appsi_highs",

                "Optimization_Scope":
                    "ELIGIBLE_ABC_RECORDS_ONLY",

                "Priority_Semantics":
                    (
                        "STRICT ECONOMIC ABC "
                        "LEXICOGRAPHIC PRIORITY"
                    ),

                "Stage_1":
                    "MAX_MIN_A",

                "Stage_2":
                    "PRESERVE_A_MAX_A_VALUE",

                "Stage_3":
                    "PRESERVE_A_MAX_MIN_B",

                "Stage_4":
                    "PRESERVE_B_MAX_B_VALUE",

                "Stage_5":
                    "PRESERVE_AB_MAX_MIN_C",

                "Stage_6":
                    "PRESERVE_C_MAX_C_VALUE",

                "Stage_7":
                    "TECHNICAL_KEY_TIE_BREAK",

                "Clinical_Priority":
                    False,

                "Actual_CDH_Budget":
                    False,

                "High_Priority_AB_Unresolved_Count":
                    unresolved_ab,

                "Full_Hospital_Claim":
                    False,

                "Automatic_Purchase_Order":
                    False,

                "Automatic_Publication":
                    False,

                "Publication_Status":
                    "CANDIDATE_NOT_PUBLISHED",

                "Run_ID":
                    RUNTIME_CONTEXT.run_id,

                "Run_Mode":
                    RUNTIME_CONTEXT.run_mode,
            }
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    allocations,
    summary,
    by_abc,
    audit,
    monotonicity,
    repeatability,
    governance,
):

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

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
            "audit_df",
            audit,
        )

        connection.register(
            "mono_df",
            monotonicity,
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
            f"""
            CREATE OR REPLACE TABLE
            {ALLOCATIONS_TABLE} AS
            SELECT * FROM alloc_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {SUMMARY_TABLE} AS
            SELECT * FROM summary_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {BY_ABC_TABLE} AS
            SELECT * FROM abc_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {PRIORITY_AUDIT_TABLE} AS
            SELECT * FROM audit_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {MONOTONICITY_TABLE} AS
            SELECT * FROM mono_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {REPEATABILITY_TABLE} AS
            SELECT * FROM repeat_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {GOVERNANCE_TABLE} AS
            SELECT * FROM gov_df
            """
        )

    finally:
        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 118)
    print("CDH INVENTORY DSS")
    print("STRICT A>B>C LEXICOGRAPHIC PROCUREMENT MILP — V2")
    print("=" * 118)
    print()

    source = load_inputs()

    validate_inputs(
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
        "Requirement policies:",
        ", ".join(
            SCENARIOS.keys()
        ),
    )

    print(
        "Budget levels:",
        "60%, 70%, 80%, 90%",
    )

    print(
        "Lexicographic stages per case: 7"
    )

    print()

    allocations = []
    summaries = []

    for scenario, requirement_column in (
        SCENARIOS.items()
    ):

        for factor in BUDGET_FACTORS:

            print("-" * 118)

            print(
                "Solving strict policy:",
                scenario,
                f"@ {factor:.0%} B100",
            )

            allocation, summary = (
                solve_case(
                    dataframe=source,
                    scenario=scenario,
                    requirement_column=(
                        requirement_column
                    ),
                    budget_factor=factor,
                )
            )

            validate_case(
                allocation,
                summary,
            )

            allocations.append(
                allocation
            )

            summaries.append(
                summary
            )

            print(
                "  A min coverage:",
                f"{summary['Observed_Min_A'] * 100:,.4f}%",
            )

            print(
                "  A allocated value:",
                f"{summary['Final_A_Value']:,.2f}",
            )

            print(
                "  B min coverage:",
                f"{summary['Observed_Min_B'] * 100:,.4f}%",
            )

            print(
                "  B allocated value:",
                f"{summary['Final_B_Value']:,.2f}",
            )

            print(
                "  C min coverage:",
                f"{summary['Observed_Min_C'] * 100:,.4f}%",
            )

            print(
                "  C allocated value:",
                f"{summary['Final_C_Value']:,.2f}",
            )

            print(
                "  Budget utilization:",
                f"{summary['Budget_Utilization_Pct']:,.6f}%",
            )

    allocation_df = pd.concat(
        allocations,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summaries
    )

    by_abc = build_by_abc(
        allocation_df
    )

    audit = build_priority_audit(
        summary_df
    )

    print()
    print(
        "Strict ABC priority audit: PASS"
    )

    monotonicity = (
        validate_monotonicity(
            summary_df
        )
    )

    print(
        "Budget monotonicity: PASS"
    )

    first_repeat = (
        allocation_df.loc[
            allocation_df[
                "Requirement_Scenario"
            ].eq(
                REPEATABILITY_SCENARIO
            )
            &
            np.isclose(
                allocation_df[
                    "Budget_Factor"
                ],
                REPEATABILITY_FACTOR,
            )
        ]
        .copy()
    )

    print(
        "Running strict POINT @ 80% "
        "repeatability check..."
    )

    repeatability = (
        run_repeatability(
            source,
            first_repeat,
        )
    )

    print(
        "Strict allocation repeatability: PASS"
    )

    governance = build_governance(
        source
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    allocation_df.to_parquet(
        OUTPUT_DIR
        / "strict_milp_allocations.parquet",
        index=False,
        engine="pyarrow",
    )

    allocation_df.to_csv(
        OUTPUT_DIR
        / "strict_milp_allocations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "strict_milp_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "strict_milp_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        OUTPUT_DIR
        / "strict_priority_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    monotonicity.to_csv(
        OUTPUT_DIR
        / "strict_budget_monotonicity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeatability.to_csv(
        OUTPUT_DIR
        / "strict_repeatability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "strict_milp_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        allocation_df,
        summary_df,
        by_abc,
        audit,
        monotonicity,
        repeatability,
        governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 118)
    print("STRICT ABC OPTIMIZATION SUMMARY")
    print("=" * 118)

    print(
        summary_df[
            [
                "Requirement_Scenario",
                "Budget_Percent",
                "Scenario_Budget",
                "Allocated_Value",
                "Unspent_Budget",
                "Budget_Utilization_Pct",
                "Observed_Min_A",
                "Final_A_Value",
                "Observed_Min_B",
                "Final_B_Value",
                "Observed_Min_C",
                "Final_C_Value",
                "Required_Units",
                "Allocated_Units",
                "Unfunded_Units",
            ]
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
    print("=" * 118)
    print("STRICT ABC COVERAGE")
    print("=" * 118)

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
    print("=" * 118)
    print("STRICT PRIORITY AUDIT")
    print("=" * 118)

    print(
        audit.to_string(
            index=False
        )
    )

    print()
    print("=" * 118)
    print("REPEATABILITY")
    print("=" * 118)

    print(
        repeatability.to_string(
            index=False
        )
    )

    print()
    print("=" * 118)
    print("GOVERNANCE")
    print("=" * 118)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 118)
    print("INTERPRETATION")
    print("-" * 118)

    print(
        "The V2 model treats A>B>C as "
        "strict economic priority."
    )

    print(
        "Within each class, minimum "
        "proportional fulfillment is "
        "maximized first."
    )

    print(
        "After preserving that fairness "
        "level, the model maximizes "
        "allocated economic value in the "
        "same class before proceeding "
        "to the next class."
    )

    print(
        "Any residual lower-class allocation "
        "while a higher class is below 100% "
        "can therefore occur only after the "
        "maximum feasible higher-class value "
        "has been preserved under the "
        "integer and budget constraints."
    )

    print(
        "ABC remains economic priority, "
        "not clinical criticality."
    )

    print(
        "These remain modeled budget "
        "scenarios, not actual CDH budgets."
    )

    unresolved_ab_count = int(
        governance[
            "High_Priority_AB_Unresolved_Count"
        ].iloc[0]
    )

    print(
        f"{unresolved_ab_count:,} unresolved "
        "A/B planning-input records remain "
        "outside the eligible optimization "
        "population."
    )

    print()
    print("=" * 118)
    print(
        "STRICT ABC MILP V2 COMPLETE"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()