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
    / "retrospective"
    / "retrospective_2025_simulation_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

SIMULATION_VERSION = (
    "RETROSPECTIVE_2025_SIMULATION_V1"
)

SCENARIOS = {
    "POINT":
        "Retrospective_Net_Requirement_POINT",

    "P80_FLOORED_CEIL":
        "Retrospective_Net_Requirement_P80_FLOORED_CEIL",

    "P90_FLOORED_CEIL":
        "Retrospective_Net_Requirement_P90_FLOORED_CEIL",
}

BUDGET_FACTORS = [
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

ABC_CLASSES = [
    "A",
    "B",
    "C",
]

EXPECTED_TOTAL_MEDICINES = 2733
EXPECTED_ELIGIBLE = 2197

THETA_TOLERANCE = 1e-7
VALUE_TOLERANCE = 0.01
BUDGET_TOLERANCE = 0.05
COVERAGE_TOLERANCE = 1e-5

REPEATABILITY_SCENARIO = "POINT"
REPEATABILITY_FACTOR = 0.80


# ============================================================
# MODEL INPUT COLUMNS
#
# Actual 2025 issuance is deliberately absent.
# ============================================================

MODEL_BASE_COLUMNS = [
    "Medicine_Key",
    "Brand_Name",
    "Generic_Name",
    "Unit",
    "ABC_AsOf",
    "Behavior_Segment_AsOf",
    "Planning_Cost",
    "Planning_Stock_Qty",
]


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
            """
            SELECT *
            FROM retrospective_2025_inputs
            ORDER BY Medicine_Key
            """
        ).df()

    finally:

        connection.close()

    return dataframe


# ============================================================
# VALIDATE TEMPORAL / STRUCTURAL INPUT
# ============================================================

def validate_inputs(
    dataframe: pd.DataFrame,
) -> None:

    if len(dataframe) != EXPECTED_TOTAL_MEDICINES:

        raise ValueError(
            f"Expected {EXPECTED_TOTAL_MEDICINES:,} "
            f"rows; found {len(dataframe):,}"
        )

    if dataframe[
        "Medicine_Key"
    ].duplicated().any():

        raise ValueError(
            "Duplicate Medicine_Key."
        )

    if not dataframe[
        "Target_Year"
    ].eq(2025).all():

        raise ValueError(
            "Unexpected target year."
        )

    if not dataframe[
        "Feature_Cutoff_Year"
    ].eq(2024).all():

        raise ValueError(
            "Feature cutoff is not 2024."
        )

    if not dataframe[
        "Max_History_Year_Used"
    ].eq(2024).all():

        raise ValueError(
            "Future history detected."
        )

    if dataframe[
        "Observed_2025_Used_For_Planning"
    ].any():

        raise ValueError(
            "2025 outcome leakage flag TRUE."
        )

    eligible = dataframe[
        "Retrospective_Optimization_Eligible"
    ].fillna(False).astype(bool)

    if int(
        eligible.sum()
    ) != EXPECTED_ELIGIBLE:

        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE:,} "
            f"eligible rows; found "
            f"{int(eligible.sum()):,}"
        )

    if not dataframe.loc[
        eligible,
        "ABC_AsOf",
    ].isin(
        ABC_CLASSES
    ).all():

        raise ValueError(
            "Non-A/B/C record in "
            "eligible population."
        )

    if dataframe.loc[
        eligible,
        "Planning_Cost",
    ].isna().any():

        raise ValueError(
            "Eligible row has missing cost."
        )

    if dataframe.loc[
        eligible,
        "Planning_Cost",
    ].le(0).any():

        raise ValueError(
            "Eligible row has invalid cost."
        )

    if dataframe.loc[
        eligible,
        "Planning_Stock_Qty",
    ].isna().any():

        raise ValueError(
            "Eligible row has missing stock."
        )

    stock_values = dataframe.loc[
        eligible,
        "Planning_Stock_Qty",
    ].astype(float).to_numpy()

    if not np.allclose(
        stock_values,
        np.round(stock_values),
        atol=1e-9,
        rtol=0,
    ):

        raise ValueError(
            "Eligible baseline stock "
            "is not whole-unit."
        )

    actual = dataframe[
        "Actual_2025_Annual_Issuance"
    ].astype(float).to_numpy()

    if (
        actual < 0
    ).any():

        raise ValueError(
            "Negative actual issuance."
        )

    if not np.allclose(
        actual,
        np.round(actual),
        atol=1e-9,
        rtol=0,
    ):

        raise ValueError(
            "Actual issuance is not "
            "whole-unit."
        )

    for column in SCENARIOS.values():

        values = dataframe.loc[
            eligible,
            column,
        ]

        if values.isna().any():

            raise ValueError(
                f"Missing {column}."
            )

        numeric = values.astype(
            float
        ).to_numpy()

        if (
            numeric < 0
        ).any():

            raise ValueError(
                f"Negative {column}."
            )

        if not np.allclose(
            numeric,
            np.round(numeric),
            atol=1e-9,
            rtol=0,
        ):

            raise ValueError(
                f"{column} is not whole-unit."
            )


# ============================================================
# BUILD MODEL
#
# This function receives only planning-origin fields.
# Actual 2025 issuance is not supplied.
# ============================================================

def build_model(
    model_data: pd.DataFrame,
    requirement_column: str,
    budget_factor: float,
):

    keys = model_data[
        "Medicine_Key"
    ].tolist()

    costs = dict(
        zip(
            keys,
            model_data[
                "Planning_Cost"
            ].astype(float),
        )
    )

    requirements = dict(
        zip(
            keys,
            model_data[
                requirement_column
            ].astype(int),
        )
    )

    abc = dict(
        zip(
            keys,
            model_data[
                "ABC_AsOf"
            ],
        )
    )

    sorted_keys = sorted(
        keys
    )

    tie_weights = {
        key:
            len(sorted_keys) - rank
        for rank, key
        in enumerate(
            sorted_keys
        )
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
        for abc_class in ABC_CLASSES
    }

    class_all = {
        abc_class: [
            key
            for key in keys
            if abc[key] == abc_class
        ]
        for abc_class in ABC_CLASSES
    }

    model = ConcreteModel()

    model.I = Set(
        initialize=keys,
        ordered=True,
    )

    model.R = Param(
        model.I,
        initialize=requirements,
        within=NonNegativeIntegers,
    )

    model.C = Param(
        model.I,
        initialize=costs,
    )

    model.W = Param(
        model.I,
        initialize=tie_weights,
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
    # SETS
    # --------------------------------------------------------

    model.A_POS = Set(
        initialize=positive_by_class["A"]
    )

    model.B_POS = Set(
        initialize=positive_by_class["B"]
    )

    model.C_POS = Set(
        initialize=positive_by_class["C"]
    )

    model.A_ALL = Set(
        initialize=class_all["A"]
    )

    model.B_ALL = Set(
        initialize=class_all["B"]
    )

    model.C_ALL = Set(
        initialize=class_all["C"]
    )

    # --------------------------------------------------------
    # COVERAGE VARIABLES
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

    if not positive_by_class["A"]:
        model.theta_A.fix(1.0)

    if not positive_by_class["B"]:
        model.theta_B.fix(1.0)

    if not positive_by_class["C"]:
        model.theta_C.fix(1.0)

    # --------------------------------------------------------
    # MINIMUM COVERAGE
    # --------------------------------------------------------

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
        model.A_POS,
        rule=coverage_a_rule,
    )

    model.coverage_B = Constraint(
        model.B_POS,
        rule=coverage_b_rule,
    )

    model.coverage_C = Constraint(
        model.C_POS,
        rule=coverage_c_rule,
    )

    # --------------------------------------------------------
    # CLASS ECONOMIC VALUE
    # --------------------------------------------------------

    model.value_A = Expression(
        expr=sum(
            model.C[i]
            *
            model.x[i]
            for i in model.A_ALL
        )
    )

    model.value_B = Expression(
        expr=sum(
            model.C[i]
            *
            model.x[i]
            for i in model.B_ALL
        )
    )

    model.value_C = Expression(
        expr=sum(
            model.C[i]
            *
            model.x[i]
            for i in model.C_ALL
        )
    )

    # --------------------------------------------------------
    # DETERMINISTIC TECHNICAL TIE-BREAK
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
    # SEVEN OBJECTIVES — SAME FROZEN V2 POLICY
    # --------------------------------------------------------

    model.obj_theta_A = Objective(
        expr=model.theta_A,
        sense=maximize,
    )

    model.obj_value_A = Objective(
        expr=model.value_A,
        sense=maximize,
    )

    model.obj_theta_B = Objective(
        expr=model.theta_B,
        sense=maximize,
    )

    model.obj_value_B = Objective(
        expr=model.value_B,
        sense=maximize,
    )

    model.obj_theta_C = Objective(
        expr=model.theta_C,
        sense=maximize,
    )

    model.obj_value_C = Objective(
        expr=model.value_C,
        sense=maximize,
    )

    model.obj_tie = Objective(
        expr=model.tie_score,
        sense=maximize,
    )

    model.obj_value_A.deactivate()
    model.obj_theta_B.deactivate()
    model.obj_value_B.deactivate()
    model.obj_theta_C.deactivate()
    model.obj_value_C.deactivate()
    model.obj_tie.deactivate()

    return (
        model,
        b100,
        budget,
    )


# ============================================================
# SOLVER
# ============================================================

def make_solver(
    deterministic: bool = False,
):

    solver = SolverFactory(
        "appsi_highs"
    )

    if not solver.available():

        raise RuntimeError(
            "appsi_highs unavailable."
        )

    if deterministic:

        solver.highs_options[
            "parallel"
        ] = "off"

        solver.highs_options[
            "threads"
        ] = 1

        solver.highs_options[
            "random_seed"
        ] = 0

    return solver


def solve_optimal(
    solver,
    model,
    stage_name: str,
):

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
# STRICT V2 SOLVE
# ============================================================

def solve_case(
    eligible_data: pd.DataFrame,
    scenario: str,
    requirement_column: str,
    budget_factor: float,
    run_technical_tiebreak: bool = True,
    deterministic_solver: bool = False,
):

    # --------------------------------------------------------
    # HARD TEMPORAL SEPARATION
    #
    # Only the following fields can enter
    # the optimization model.
    # --------------------------------------------------------

    model_columns = (
        MODEL_BASE_COLUMNS
        +
        [
            requirement_column,
        ]
    )

    model_data = eligible_data[
        model_columns
    ].copy()

    if (
        "Actual_2025_Annual_Issuance"
        in
        model_data.columns
    ):

        raise RuntimeError(
            "Actual 2025 issuance entered "
            "the optimization model."
        )

    model, b100, budget = (
        build_model(
            model_data=model_data,
            requirement_column=(
                requirement_column
            ),
            budget_factor=(
                budget_factor
            ),
        )
    )

    solver = make_solver(
        deterministic=(
            deterministic_solver
        )
    )

    # ========================================================
    # STAGE 1 — MAX MINIMUM A
    # ========================================================

    solve_optimal(
        solver,
        model,
        "Stage 1 theta A",
    )

    theta_a_star = float(
        value(
            model.theta_A
        )
    )

    model.keep_theta_A = Constraint(
        expr=(
            model.theta_A
            >=
            max(
                0.0,
                theta_a_star
                -
                THETA_TOLERANCE,
            )
        )
    )

    model.obj_theta_A.deactivate()
    model.obj_value_A.activate()

    # ========================================================
    # STAGE 2 — MAXIMUM A VALUE
    # ========================================================

    solve_optimal(
        solver,
        model,
        "Stage 2 A value",
    )

    value_a_star = float(
        value(
            model.value_A
        )
    )

    model.keep_value_A = Constraint(
        expr=(
            model.value_A
            >=
            value_a_star
            -
            VALUE_TOLERANCE
        )
    )

    model.obj_value_A.deactivate()
    model.obj_theta_B.activate()

    # ========================================================
    # STAGE 3 — MAX MINIMUM B
    # ========================================================

    solve_optimal(
        solver,
        model,
        "Stage 3 theta B",
    )

    theta_b_star = float(
        value(
            model.theta_B
        )
    )

    model.keep_theta_B = Constraint(
        expr=(
            model.theta_B
            >=
            max(
                0.0,
                theta_b_star
                -
                THETA_TOLERANCE,
            )
        )
    )

    model.obj_theta_B.deactivate()
    model.obj_value_B.activate()

    # ========================================================
    # STAGE 4 — MAXIMUM B VALUE
    # ========================================================

    solve_optimal(
        solver,
        model,
        "Stage 4 B value",
    )

    value_b_star = float(
        value(
            model.value_B
        )
    )

    model.keep_value_B = Constraint(
        expr=(
            model.value_B
            >=
            value_b_star
            -
            VALUE_TOLERANCE
        )
    )

    model.obj_value_B.deactivate()
    model.obj_theta_C.activate()

    # ========================================================
    # STAGE 5 — MAX MINIMUM C
    # ========================================================

    solve_optimal(
        solver,
        model,
        "Stage 5 theta C",
    )

    theta_c_star = float(
        value(
            model.theta_C
        )
    )

    model.keep_theta_C = Constraint(
        expr=(
            model.theta_C
            >=
            max(
                0.0,
                theta_c_star
                -
                THETA_TOLERANCE,
            )
        )
    )

    model.obj_theta_C.deactivate()
    model.obj_value_C.activate()

    # ========================================================
    # STAGE 6 — MAXIMUM C VALUE
    # ========================================================

    solve_optimal(
        solver,
        model,
        "Stage 6 C value",
    )

    value_c_star = float(
        value(
            model.value_C
        )
    )

    if run_technical_tiebreak:

        model.keep_value_C = Constraint(
            expr=(
                model.value_C
                >=
                value_c_star
                -
                VALUE_TOLERANCE
            )
        )

        model.obj_value_C.deactivate()
        model.obj_tie.activate()

        # ====================================================
        # STAGE 7 — OPTIONAL TECHNICAL TIE-BREAK
        # ====================================================

        solve_optimal(
            solver,
            model,
            "Stage 7 tie-break",
        )

    # --------------------------------------------------------
    # EXTRACT ALLOCATION BEFORE OUTCOME EVALUATION
    # --------------------------------------------------------

    allocation = eligible_data[
        [
            "Medicine_Key",
            "Brand_Name",
            "Generic_Name",
            "Unit",
            "ABC_AsOf",
            "Behavior_Segment_AsOf",
            "Planning_Cost",
            "Planning_Stock_Qty",
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
    ] = eligible_data[
        requirement_column
    ].astype(
        np.int64
    ).to_numpy()

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
        for key in eligible_data[
            "Medicine_Key"
        ]
    ]

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
        "Unfunded_Requirement_Value"
    ] = (
        allocation[
            "Unfunded_Requirement_Units"
        ]
        *
        allocation[
            "Planning_Cost"
        ]
    )

    # ========================================================
    # ONLY NOW REVEAL ACTUAL 2025 OUTCOME
    # ========================================================

    outcome = eligible_data[
        [
            "Medicine_Key",
            "Actual_2025_Annual_Issuance",
        ]
    ].copy()

    allocation = allocation.merge(
        outcome,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    allocation[
        "Actual_2025_Annual_Issuance"
    ] = (
        allocation[
            "Actual_2025_Annual_Issuance"
        ]
        .astype(
            np.int64
        )
    )

    allocation[
        "Baseline_Stock_Units"
    ] = (
        np.round(
            allocation[
                "Planning_Stock_Qty"
            ]
        )
        .astype(
            np.int64
        )
    )

    allocation[
        "Simulated_Available_Units"
    ] = (
        allocation[
            "Baseline_Stock_Units"
        ]
        +
        allocation[
            "Allocated_Units"
        ]
    )

    allocation[
        "Stock_Only_Simulated_Shortfall_Units"
    ] = np.maximum(
        0,
        (
            allocation[
                "Actual_2025_Annual_Issuance"
            ]
            -
            allocation[
                "Baseline_Stock_Units"
            ]
        ),
    )

    allocation[
        "Stock_Only_Simulated_Residual_Units"
    ] = np.maximum(
        0,
        (
            allocation[
                "Baseline_Stock_Units"
            ]
            -
            allocation[
                "Actual_2025_Annual_Issuance"
            ]
        ),
    )

    allocation[
        "Simulated_Annual_Issuance_Shortfall_Units"
    ] = np.maximum(
        0,
        (
            allocation[
                "Actual_2025_Annual_Issuance"
            ]
            -
            allocation[
                "Simulated_Available_Units"
            ]
        ),
    )

    allocation[
        "Simulated_Residual_Inventory_Units"
    ] = np.maximum(
        0,
        (
            allocation[
                "Simulated_Available_Units"
            ]
            -
            allocation[
                "Actual_2025_Annual_Issuance"
            ]
        ),
    )

    allocation[
        "Simulated_Issuance_Covered_Units"
    ] = np.minimum(
        allocation[
            "Actual_2025_Annual_Issuance"
        ],
        allocation[
            "Simulated_Available_Units"
        ],
    )

    allocation[
        "Simulated_Shortfall_Reduction_vs_Stock_Only"
    ] = (
        allocation[
            "Stock_Only_Simulated_Shortfall_Units"
        ]
        -
        allocation[
            "Simulated_Annual_Issuance_Shortfall_Units"
        ]
    )

    allocation[
        "Simulated_Residual_Increase_vs_Stock_Only"
    ] = (
        allocation[
            "Simulated_Residual_Inventory_Units"
        ]
        -
        allocation[
            "Stock_Only_Simulated_Residual_Units"
        ]
    )

    allocation[
        "Simulated_Shortfall_Value_At_Planning_Cost"
    ] = (
        allocation[
            "Simulated_Annual_Issuance_Shortfall_Units"
        ]
        *
        allocation[
            "Planning_Cost"
        ]
    )

    allocation[
        "Simulated_Residual_Value_At_Planning_Cost"
    ] = (
        allocation[
            "Simulated_Residual_Inventory_Units"
        ]
        *
        allocation[
            "Planning_Cost"
        ]
    )

    allocation[
        "Simulation_Is_Actual_Stockout"
    ] = False

    allocation[
        "Simulation_Is_Actual_Waste"
    ] = False

    allocation[
        "Simulation_Version"
    ] = SIMULATION_VERSION

    # --------------------------------------------------------
    # CLASS METRICS
    # --------------------------------------------------------

    class_metrics = {}

    for abc_class in ABC_CLASSES:

        group = allocation.loc[
            allocation[
                "ABC_AsOf"
            ].eq(
                abc_class
            )
        ]

        positive_requirement = group.loc[
            group[
                "Net_Requirement"
            ].gt(0)
        ]

        if positive_requirement.empty:

            minimum_coverage = 1.0

        else:

            minimum_coverage = float(
                (
                    positive_requirement[
                        "Allocated_Units"
                    ]
                    /
                    positive_requirement[
                        "Net_Requirement"
                    ]
                )
                .min()
            )

        class_metrics[
            abc_class
        ] = {
            "allocated_value":
                float(
                    group[
                        "Allocated_Value"
                    ].sum()
                ),

            "minimum_coverage":
                minimum_coverage,
        }

    allocated_value = float(
        allocation[
            "Allocated_Value"
        ].sum()
    )

    actual_units = int(
        allocation[
            "Actual_2025_Annual_Issuance"
        ].sum()
    )

    covered_units = int(
        allocation[
            "Simulated_Issuance_Covered_Units"
        ].sum()
    )

    shortfall_units = int(
        allocation[
            "Simulated_Annual_Issuance_Shortfall_Units"
        ].sum()
    )

    residual_units = int(
        allocation[
            "Simulated_Residual_Inventory_Units"
        ].sum()
    )

    stock_only_shortfall = int(
        allocation[
            "Stock_Only_Simulated_Shortfall_Units"
        ].sum()
    )

    summary = {
        "Requirement_Scenario":
            scenario,

        "Budget_Factor":
            budget_factor,

        "Budget_Percent":
            budget_factor * 100,

        "Eligible_Medicine_Count":
            len(
                allocation
            ),

        "B100":
            b100,

        "Scenario_Budget":
            budget,

        "Allocated_Value":
            allocated_value,

        "Unspent_Budget":
            (
                budget
                -
                allocated_value
            ),

        "Budget_Utilization_Pct":
            (
                allocated_value
                /
                budget
                *
                100
                if budget > 0
                else 100.0
            ),

        "Theta_A_Star":
            theta_a_star,

        "A_Value_Star":
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

        "B_Value_Star":
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

        "C_Value_Star":
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

        "Actual_2025_Issuance_Eligible_Units":
            actual_units,

        "Stock_Only_Simulated_Shortfall_Units":
            stock_only_shortfall,

        "Simulated_Issuance_Covered_Units":
            covered_units,

        "Simulated_Annual_Issuance_Shortfall_Units":
            shortfall_units,

        "Simulated_Residual_Inventory_Units":
            residual_units,

        "Simulated_Fulfillment_Pct":
            (
                covered_units
                /
                actual_units
                *
                100
                if actual_units > 0
                else 100.0
            ),

        "Simulated_Shortfall_Medicine_Count":
            int(
                allocation[
                    "Simulated_Annual_Issuance_Shortfall_Units"
                ]
                .gt(0)
                .sum()
            ),

        "Simulated_Residual_Medicine_Count":
            int(
                allocation[
                    "Simulated_Residual_Inventory_Units"
                ]
                .gt(0)
                .sum()
            ),

        "Shortfall_Reduction_vs_Stock_Only_Units":
            int(
                stock_only_shortfall
                -
                shortfall_units
            ),

        "Simulated_Shortfall_Value_At_Planning_Cost":
            float(
                allocation[
                    "Simulated_Shortfall_Value_At_Planning_Cost"
                ].sum()
            ),

        "Simulated_Residual_Value_At_Planning_Cost":
            float(
                allocation[
                    "Simulated_Residual_Value_At_Planning_Cost"
                ].sum()
            ),

        "Actual_2025_Used_In_Optimization":
            False,

        "Simulation_Version":
            SIMULATION_VERSION,
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
            "A coverage preservation failed."
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
            "B coverage preservation failed."
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
            "C coverage preservation failed."
        )

    if (
        summary[
            "Final_A_Value"
        ]
        <
        summary[
            "A_Value_Star"
        ]
        -
        VALUE_TOLERANCE
        -
        1e-6
    ):

        raise RuntimeError(
            "A value preservation failed."
        )

    if (
        summary[
            "Final_B_Value"
        ]
        <
        summary[
            "B_Value_Star"
        ]
        -
        VALUE_TOLERANCE
        -
        1e-6
    ):

        raise RuntimeError(
            "B value preservation failed."
        )

    if (
        summary[
            "Final_C_Value"
        ]
        <
        summary[
            "C_Value_Star"
        ]
        -
        VALUE_TOLERANCE
        -
        1e-6
    ):

        raise RuntimeError(
            "C value preservation failed."
        )

    if summary[
        "Actual_2025_Used_In_Optimization"
    ]:

        raise RuntimeError(
            "Actual 2025 outcome leakage."
        )

    # --------------------------------------------------------
    # B100 SANITY CHECK
    # --------------------------------------------------------

    if np.isclose(
        summary[
            "Budget_Factor"
        ],
        1.0,
    ):

        mismatches = int(
            (
                allocation[
                    "Allocated_Units"
                ]
                !=
                allocation[
                    "Net_Requirement"
                ]
            )
            .sum()
        )

        if mismatches != 0:

            raise RuntimeError(
                "100% retrospective case "
                "did not fully fund all "
                "eligible net requirements."
            )

        for column in [
            "Observed_Min_A",
            "Observed_Min_B",
            "Observed_Min_C",
        ]:

            if abs(
                summary[
                    column
                ]
                -
                1.0
            ) > COVERAGE_TOLERANCE:

                raise RuntimeError(
                    "100% retrospective "
                    "coverage is not 1.0."
                )


# ============================================================
# BY ABC SIMULATION SUMMARY
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

        actual = int(
            group[
                "Actual_2025_Annual_Issuance"
            ].sum()
        )

        covered = int(
            group[
                "Simulated_Issuance_Covered_Units"
            ].sum()
        )

        shortfall = int(
            group[
                "Simulated_Annual_Issuance_Shortfall_Units"
            ].sum()
        )

        residual = int(
            group[
                "Simulated_Residual_Inventory_Units"
            ].sum()
        )

        rows.append(
            {
                "Requirement_Scenario":
                    scenario,

                "Budget_Factor":
                    factor,

                "Budget_Percent":
                    factor
                    *
                    100,

                "ABC_AsOf":
                    abc_class,

                "Medicine_Count":
                    len(
                        group
                    ),

                "Actual_2025_Issuance_Units":
                    actual,

                "Allocated_Procurement_Units":
                    int(
                        group[
                            "Allocated_Units"
                        ].sum()
                    ),

                "Allocated_Procurement_Value":
                    float(
                        group[
                            "Allocated_Value"
                        ].sum()
                    ),

                "Simulated_Covered_Issuance_Units":
                    covered,

                "Simulated_Annual_Issuance_Shortfall_Units":
                    shortfall,

                "Simulated_Residual_Inventory_Units":
                    residual,

                "Simulated_Fulfillment_Pct":
                    (
                        covered
                        /
                        actual
                        *
                        100
                        if actual > 0
                        else 100.0
                    ),

                "Shortfall_Medicine_Count":
                    int(
                        group[
                            "Simulated_Annual_Issuance_Shortfall_Units"
                        ]
                        .gt(0)
                        .sum()
                    ),

                "Residual_Medicine_Count":
                    int(
                        group[
                            "Simulated_Residual_Inventory_Units"
                        ]
                        .gt(0)
                        .sum()
                    ),

                "Simulation_Version":
                    SIMULATION_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ELIGIBLE / EXCLUDED OUTCOME COVERAGE
# ============================================================

def build_population_summary(
    full_data: pd.DataFrame,
) -> pd.DataFrame:

    eligible = full_data[
        "Retrospective_Optimization_Eligible"
    ].fillna(False).astype(bool)

    total_actual = float(
        full_data[
            "Actual_2025_Annual_Issuance"
        ].sum()
    )

    eligible_actual = float(
        full_data.loc[
            eligible,
            "Actual_2025_Annual_Issuance",
        ].sum()
    )

    excluded_actual = float(
        full_data.loc[
            ~eligible,
            "Actual_2025_Annual_Issuance",
        ].sum()
    )

    return pd.DataFrame(
        [
            {
                "Total_Medicine_Count":
                    len(
                        full_data
                    ),

                "Eligible_Medicine_Count":
                    int(
                        eligible.sum()
                    ),

                "Excluded_Medicine_Count":
                    int(
                        (
                            ~eligible
                        ).sum()
                    ),

                "Total_Actual_2025_Issuance":
                    total_actual,

                "Eligible_Actual_2025_Issuance":
                    eligible_actual,

                "Excluded_Actual_2025_Issuance":
                    excluded_actual,

                "Eligible_Actual_Issuance_Share_Pct":
                    (
                        eligible_actual
                        /
                        total_actual
                        *
                        100
                        if total_actual > 0
                        else np.nan
                    ),

                "High_Priority_AB_Unresolved_Count":
                    int(
                        full_data[
                            "High_Priority_AB_Unresolved"
                        ].sum()
                    ),

                "Simulation_Scope":
                    (
                        "ELIGIBLE_ABC_RECORDS_ONLY"
                    ),

                "Simulation_Version":
                    SIMULATION_VERSION,
            }
        ]
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
    eligible_data: pd.DataFrame,
    first_solution: pd.DataFrame,
) -> pd.DataFrame:

    second_solution, second_summary = (
        solve_case(
            eligible_data=eligible_data,
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
        second_solution,
        second_summary,
    )

    first = first_solution.sort_values(
        "Medicine_Key"
    ).reset_index(
        drop=True
    )

    second = second_solution.sort_values(
        "Medicine_Key"
    ).reset_index(
        drop=True
    )

    mismatch_count = int(
        (
            first[
                "Allocated_Units"
            ].to_numpy()
            !=
            second[
                "Allocated_Units"
            ].to_numpy()
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
        mismatch_count == 0
        and
        first_hash == second_hash
    )

    if not passed:

        raise RuntimeError(
            "Retrospective repeatability "
            "check failed."
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
                    mismatch_count,

                "First_SHA256":
                    first_hash,

                "Second_SHA256":
                    second_hash,

                "Exact_Repeatability_PASS":
                    passed,

                "Simulation_Version":
                    SIMULATION_VERSION,
            }
        ]
    )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "Simulation_Version":
                    SIMULATION_VERSION,

                "Planning_Origin":
                    "END_OF_2024",

                "Target_Year":
                    2025,

                "Pyomo_Version":
                    pyomo.__version__,

                "HiGHS_Version":
                    highspy.Highs().version(),

                "Solver_Interface":
                    "appsi_highs",

                "Optimizer_Policy":
                    (
                        "FROZEN_STRICT_ABC_V2"
                    ),

                "Budget_Levels":
                    "60%,70%,80%,90%,100%",

                "Budget_Levels_Are_Actual_CDH_Budget":
                    False,

                "Actual_2025_Issuance_Entered_Model":
                    False,

                "Actual_2025_Issuance_Used_For_Evaluation":
                    True,

                "Outcome_Shortfall_Name":
                    (
                        "SIMULATED_ANNUAL_"
                        "ISSUANCE_SHORTFALL"
                    ),

                "Outcome_Residual_Name":
                    (
                        "SIMULATED_RESIDUAL_"
                        "INVENTORY"
                    ),

                "Actual_Stockout_Claim":
                    False,

                "Actual_Waste_Claim":
                    False,

                "Within_Year_Timing_Modelled":
                    False,

                "Full_Hospital_Claim":
                    False,

                "Reason_Full_Claim_Blocked":
                    (
                        "Four A/B medicines were "
                        "unresolved at the end-of-2024 "
                        "planning origin."
                    ),
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
    population,
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
            "population_df",
            population,
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
            retrospective_2025_simulation_allocations AS
            SELECT * FROM alloc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_simulation_summary AS
            SELECT * FROM summary_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_simulation_by_abc AS
            SELECT * FROM abc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_simulation_population AS
            SELECT * FROM population_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_simulation_repeatability AS
            SELECT * FROM repeat_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_simulation_governance AS
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
    print("2025 RETROSPECTIVE PROCUREMENT + ANNUAL ISSUANCE SIMULATION")
    print("=" * 118)
    print()

    full_data = load_inputs()

    validate_inputs(
        full_data
    )

    print(
        "Retrospective temporal validation: PASS"
    )

    print(
        "Planning origin: end of 2024"
    )

    print(
        "Actual 2025 issuance enters "
        "optimization model: NO"
    )

    eligible = full_data.loc[
        full_data[
            "Retrospective_Optimization_Eligible"
        ]
    ].copy()

    print(
        "Eligible retrospective medicines:",
        f"{len(eligible):,}",
    )

    print(
        "Requirement policies:",
        ", ".join(
            SCENARIOS.keys()
        ),
    )

    print(
        "Budget levels:",
        "60%, 70%, 80%, 90%, 100%",
    )

    print(
        "Total retrospective scenarios:",
        (
            len(SCENARIOS)
            *
            len(BUDGET_FACTORS)
        ),
    )

    print()

    allocation_frames = []
    summary_rows = []

    for scenario, requirement_column in (
        SCENARIOS.items()
    ):

        for factor in BUDGET_FACTORS:

            print("-" * 118)

            print(
                "Solving retrospective:",
                scenario,
                f"@ {factor:.0%} B100",
            )

            allocation, summary = (
                solve_case(
                    eligible_data=eligible,
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

            allocation_frames.append(
                allocation
            )

            summary_rows.append(
                summary
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
                "  A min requirement coverage:",
                f"{summary['Observed_Min_A'] * 100:,.4f}%",
            )

            print(
                "  B min requirement coverage:",
                f"{summary['Observed_Min_B'] * 100:,.4f}%",
            )

            print(
                "  C min requirement coverage:",
                f"{summary['Observed_Min_C'] * 100:,.4f}%",
            )

            print(
                "  Simulated issuance shortfall:",
                f"{summary['Simulated_Annual_Issuance_Shortfall_Units']:,}",
            )

            print(
                "  Simulated residual inventory:",
                f"{summary['Simulated_Residual_Inventory_Units']:,}",
            )

            print(
                "  Simulated issuance fulfillment:",
                f"{summary['Simulated_Fulfillment_Pct']:,.4f}%",
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

    population = build_population_summary(
        full_data
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
                "Budget_Factor"
            ],
            REPEATABILITY_FACTOR,
        )
    ].copy()

    print()
    print(
        "Running retrospective POINT @ 80% "
        "repeatability check..."
    )

    repeatability = run_repeatability(
        eligible_data=eligible,
        first_solution=first_repeat,
    )

    print(
        "Retrospective allocation "
        "repeatability: PASS"
    )

    governance = build_governance()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    allocations.to_parquet(
        OUTPUT_DIR
        / "retrospective_2025_simulation_allocations.parquet",
        index=False,
        engine="pyarrow",
    )

    allocations.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_simulation_allocations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_simulation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_simulation_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    population.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_simulation_population.csv",
        index=False,
        encoding="utf-8-sig",
    )

    repeatability.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_simulation_repeatability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "retrospective_2025_simulation_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        allocations,
        summary,
        by_abc,
        population,
        repeatability,
        governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 118)
    print("RETROSPECTIVE SIMULATION POPULATION")
    print("=" * 118)

    print(
        population.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("=" * 118)
    print("RETROSPECTIVE SIMULATION SUMMARY")
    print("=" * 118)

    display_columns = [
        "Requirement_Scenario",
        "Budget_Percent",
        "Scenario_Budget",
        "Allocated_Value",
        "Budget_Utilization_Pct",
        "Actual_2025_Issuance_Eligible_Units",
        "Stock_Only_Simulated_Shortfall_Units",
        "Simulated_Annual_Issuance_Shortfall_Units",
        "Shortfall_Reduction_vs_Stock_Only_Units",
        "Simulated_Residual_Inventory_Units",
        "Simulated_Fulfillment_Pct",
        "Simulated_Shortfall_Medicine_Count",
        "Simulated_Residual_Medicine_Count",
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
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("=" * 118)
    print("RETROSPECTIVE SIMULATION BY AS-OF ABC")
    print("=" * 118)

    print(
        by_abc[
            [
                "Requirement_Scenario",
                "Budget_Percent",
                "ABC_AsOf",
                "Actual_2025_Issuance_Units",
                "Allocated_Procurement_Units",
                "Allocated_Procurement_Value",
                "Simulated_Covered_Issuance_Units",
                "Simulated_Annual_Issuance_Shortfall_Units",
                "Simulated_Residual_Inventory_Units",
                "Simulated_Fulfillment_Pct",
                "Shortfall_Medicine_Count",
                "Residual_Medicine_Count",
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
    print("REPEATABILITY")
    print("=" * 118)

    print(
        repeatability.to_string(
            index=False
        )
    )

    print()
    print("=" * 118)
    print("RETROSPECTIVE GOVERNANCE")
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
        "Actual 2025 issuance was not "
        "available to the optimization model."
    )

    print(
        "It was introduced only after "
        "procurement allocations were fixed."
    )

    print(
        "Simulated annual issuance shortfall "
        "means observed 2025 stock issuance "
        "exceeded simulated annual availability."
    )

    print(
        "It is not evidence of an actual "
        "historical stockout or unmet "
        "clinical demand."
    )

    print(
        "Simulated residual inventory is "
        "simulated availability remaining "
        "after observed annual issuance."
    )

    print(
        "It is not evidence of actual waste "
        "or expiry."
    )

    print(
        "The simulation has no within-year "
        "timing, lead-time, or replenishment "
        "sequence information."
    )

    print(
        "Budget percentages are modeled "
        "scenario budgets rather than "
        "actual CDH historical budgets."
    )

    print(
        "Cross-policy comparisons at the "
        "same percentage should be interpreted "
        "carefully because each policy has "
        "a different B100 peso amount."
    )

    print()
    print("=" * 118)
    print(
        "2025 RETROSPECTIVE SIMULATION COMPLETE"
    )
    print("=" * 118)


if __name__ == "__main__":
    main()