from __future__ import annotations

from pathlib import Path

import duckdb
import highspy
import numpy as np
import pandas as pd
import pyomo

from pyomo.environ import (
    ConcreteModel,
    Constraint,
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
        f"milp_full_funding_{TARGET_YEAR}_"
        "governed_v1"
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOCATIONS_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_full_funding_allocations"
)

VALIDATION_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_full_funding_validation"
)

BY_ABC_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_full_funding_by_abc"
)

GOVERNANCE_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_full_funding_governance"
)


# ============================================================
# GOVERNANCE
# ============================================================

VALIDATION_VERSION = (
    f"MILP_FULL_FUNDING_VALIDATION_"
    f"{TARGET_YEAR}_GOVERNED_V1"
)

SCENARIOS = {
    "POINT":
        "MILP_Net_Requirement_POINT",

    "P80_FLOORED_CEIL":
        "MILP_Net_Requirement_P80_FLOORED_CEIL",

    "P90_FLOORED_CEIL":
        "MILP_Net_Requirement_P90_FLOORED_CEIL",
}

ABC_CLASSES = [
    "A",
    "B",
    "C",
]

NUMERICAL_TOLERANCE = 1e-6


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
# VALIDATE SOURCE
# ============================================================

def validate_source(
    dataframe: pd.DataFrame,
) -> None:

    if dataframe.empty:

        raise ValueError(
            "No strict A/B/C eligible rows "
            "were available for B100 validation."
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
            "in MILP inputs."
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
            "MILP population contains "
            "a non-A/B/C record."
        )

    if (
        dataframe[
            "Planning_Cost"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Missing planning cost "
            "in eligible population."
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

    for column in (
        SCENARIOS.values()
    ):

        if (
            dataframe[
                column
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                f"Missing {column}."
            )

        values = (
            dataframe[
                column
            ]
            .astype(float)
            .to_numpy()
        )

        if (
            values < 0
        ).any():

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
                f"{column} is not "
                "whole-unit."
            )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    dataframe: pd.DataFrame,
    requirement_column: str,
) -> tuple[
    ConcreteModel,
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

    b100 = float(
        sum(
            cost[key]
            *
            requirement[key]
            for key in medicine_keys
        )
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

    model.budget = Constraint(
        expr=sum(
            model.C[i]
            *
            model.x[i]
            for i in model.I
        )
        <=
        b100
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

    # If a class had no positive
    # requirement, its coverage level
    # is conventionally fixed to 1.

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
    # LEXICOGRAPHIC OBJECTIVES
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
        expr=sum(
            model.C[i]
            *
            model.x[i]
            for i in model.I
        ),
        sense=maximize,
    )

    model.objective_B.deactivate()
    model.objective_C.deactivate()
    model.objective_value.deactivate()

    return (
        model,
        b100,
    )


# ============================================================
# SOLVER HELPER
# ============================================================

def solve_and_check(
    solver,
    model: ConcreteModel,
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
            f"{stage_name}: solver did "
            f"not terminate optimally. "
            f"Termination = {termination}"
        )

    return results


# ============================================================
# PRESERVE COVERAGE
# ============================================================

def preserved_level(
    theta_value: float,
) -> float:

    # At B100 each class should reach 1.
    # If the solver reports something
    # numerically indistinguishable from 1,
    # preserve it as exactly 1 so that
    # integer x must equal R.

    if abs(
        theta_value
        -
        1.0
    ) <= NUMERICAL_TOLERANCE:

        return 1.0

    return float(
        theta_value
    )


# ============================================================
# RUN LEXICOGRAPHIC SOLVE
# ============================================================

def solve_full_funding(
    dataframe: pd.DataFrame,
    scenario: str,
    requirement_column: str,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    model, b100 = build_model(
        dataframe,
        requirement_column,
    )

    solver = SolverFactory(
        "appsi_highs"
    )

    if not solver.available():

        raise RuntimeError(
            "appsi_highs is unavailable."
        )

    # --------------------------------------------------------
    # STAGE 1 — CATEGORY A
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        f"{scenario} Stage 1",
    )

    theta_a = float(
        value(
            model.theta_A
        )
    )

    preserve_a = preserved_level(
        theta_a
    )

    model.preserve_A = Constraint(
        expr=(
            model.theta_A
            >=
            preserve_a
        )
    )

    model.objective_A.deactivate()
    model.objective_B.activate()

    # --------------------------------------------------------
    # STAGE 2 — CATEGORY B
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        f"{scenario} Stage 2",
    )

    theta_b = float(
        value(
            model.theta_B
        )
    )

    preserve_b = preserved_level(
        theta_b
    )

    model.preserve_B = Constraint(
        expr=(
            model.theta_B
            >=
            preserve_b
        )
    )

    model.objective_B.deactivate()
    model.objective_C.activate()

    # --------------------------------------------------------
    # STAGE 3 — CATEGORY C
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        f"{scenario} Stage 3",
    )

    theta_c = float(
        value(
            model.theta_C
        )
    )

    preserve_c = preserved_level(
        theta_c
    )

    model.preserve_C = Constraint(
        expr=(
            model.theta_C
            >=
            preserve_c
        )
    )

    model.objective_C.deactivate()
    model.objective_value.activate()

    # --------------------------------------------------------
    # STAGE 4 — USE AVAILABLE BUDGET
    # --------------------------------------------------------

    solve_and_check(
        solver,
        model,
        f"{scenario} Stage 4",
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
    # EXTRACT ALLOCATION
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
        "Scenario"
    ] = scenario

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
        "Unit_Difference_x_minus_R"
    ] = (
        allocation[
            "Allocated_Units"
        ]
        -
        allocation[
            "Net_Requirement"
        ]
    )

    allocation[
        "Exact_Full_Allocation"
    ] = (
        allocation[
            "Allocated_Units"
        ]
        ==
        allocation[
            "Net_Requirement"
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
        "Validation_Version"
    ] = VALIDATION_VERSION

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    mismatch_count = int(
        (
            ~allocation[
                "Exact_Full_Allocation"
            ]
        )
        .sum()
    )

    max_abs_unit_difference = int(
        allocation[
            "Unit_Difference_x_minus_R"
        ]
        .abs()
        .max()
    )

    total_required_units = int(
        allocation[
            "Net_Requirement"
        ]
        .sum()
    )

    total_allocated_units = int(
        allocation[
            "Allocated_Units"
        ]
        .sum()
    )

    total_required_value = float(
        allocation[
            "Required_Value"
        ]
        .sum()
    )

    total_allocated_value = float(
        allocation[
            "Allocated_Value"
        ]
        .sum()
    )

    budget_difference = float(
        total_allocated_value
        -
        b100
    )

    if b100 > 0:

        budget_utilization = (
            total_allocated_value
            /
            b100
            *
            100
        )

    else:

        budget_utilization = 100.0

    summary = {
        "Scenario":
            scenario,

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

        "Total_Required_Units":
            total_required_units,

        "Total_Allocated_Units":
            total_allocated_units,

        "B100":
            b100,

        "Total_Required_Value":
            total_required_value,

        "Total_Allocated_Value":
            total_allocated_value,

        "Budget_Difference":
            budget_difference,

        "Budget_Utilization_Pct":
            budget_utilization,

        "Theta_A":
            final_theta_a,

        "Theta_B":
            final_theta_b,

        "Theta_C":
            final_theta_c,

        "Full_Allocation_Mismatch_Count":
            mismatch_count,

        "Max_Abs_Unit_Difference":
            max_abs_unit_difference,

        "Exact_Full_Allocation_PASS":
            bool(
                mismatch_count
                ==
                0
            ),

        "Theta_A_PASS":
            bool(
                abs(
                    final_theta_a
                    -
                    1.0
                )
                <=
                NUMERICAL_TOLERANCE
            ),

        "Theta_B_PASS":
            bool(
                abs(
                    final_theta_b
                    -
                    1.0
                )
                <=
                NUMERICAL_TOLERANCE
            ),

        "Theta_C_PASS":
            bool(
                abs(
                    final_theta_c
                    -
                    1.0
                )
                <=
                NUMERICAL_TOLERANCE
            ),

        "Budget_Reproduction_PASS":
            bool(
                abs(
                    total_allocated_value
                    -
                    b100
                )
                <=
                max(
                    0.01,
                    NUMERICAL_TOLERANCE
                    *
                    max(
                        1.0,
                        b100,
                    )
                )
            ),

        "Validation_VERSION":
            VALIDATION_VERSION,
    }

    return (
        allocation,
        summary,
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
        abc_class,
    ), group in allocations.groupby(
        [
            "Scenario",
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

        rows.append(
            {
                "Scenario":
                    scenario,

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

                "Unit_Coverage_Pct":
                    (
                        allocated_units
                        /
                        required_units
                        *
                        100
                        if required_units
                        > 0
                        else 100.0
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
                        if required_value
                        > 0
                        else 100.0
                    ),

                "Mismatch_Count":
                    int(
                        (
                            ~group[
                                "Exact_Full_Allocation"
                            ]
                        )
                        .sum()
                    ),

                "Validation_Version":
                    VALIDATION_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_results(
    summary: pd.DataFrame,
) -> None:

    required_pass_columns = [
        "Exact_Full_Allocation_PASS",
        "Theta_A_PASS",
        "Theta_B_PASS",
        "Theta_C_PASS",
        "Budget_Reproduction_PASS",
    ]

    for column in (
        required_pass_columns
    ):

        if not (
            summary[
                column
            ]
            .all()
        ):

            failed = (
                summary.loc[
                    ~summary[
                        column
                    ],
                    [
                        "Scenario",
                        column,
                    ],
                ]
            )

            raise RuntimeError(
                f"Full-funding validation "
                f"FAILED for {column}:\n"
                f"{failed}"
            )


# ============================================================
# GOVERNANCE
# ============================================================

def build_governance(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    highs_version = (
        highspy.Highs()
        .version()
    )

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
                "Validation_Version":
                    VALIDATION_VERSION,

                "Target_Year":
                    TARGET_YEAR,

                "Eligible_ABC_Rows":
                    len(dataframe),

                "Pyomo_Version":
                    pyomo.__version__,

                "HiGHS_Version":
                    highs_version,

                "Solver_Interface":
                    "appsi_highs",

                "Decision_Domain":
                    "NONNEGATIVE_INTEGER",

                "Budget_Case":
                    "B100_ONLY",

                "Expected_B100_Result":
                    "x_i = R_i",

                "Expected_Theta_A":
                    1.0,

                "Expected_Theta_B":
                    1.0,

                "Expected_Theta_C":
                    1.0,

                "Partial_Budget_Scenarios_Run":
                    False,

                "High_Priority_AB_Unresolved_Count":
                    unresolved_ab,

                "Full_Hospital_Claim":
                    False,

                "Reason_Full_Hospital_Claim_Blocked":
                    (
                        f"{unresolved_ab:,} high-priority "
                        "A/B planning-input records remain "
                        "outside the eligible optimization "
                        "population."
                    ),

                "Clinical_Priority_Used":
                    False,

                "Actual_CDH_Budget":
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
    allocations: pd.DataFrame,
    summary: pd.DataFrame,
    by_abc: pd.DataFrame,
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
            "governance_df",
            governance,
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {ALLOCATIONS_TABLE} AS
            SELECT *
            FROM allocations_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {VALIDATION_TABLE} AS
            SELECT *
            FROM summary_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {BY_ABC_TABLE} AS
            SELECT *
            FROM abc_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {GOVERNANCE_TABLE} AS
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
    print("100% FULL-FUNDING LEXICOGRAPHIC MILP VALIDATION")
    print("=" * 112)
    print()

    dataframe = load_inputs()

    validate_source(
        dataframe
    )

    print(
        "MILP source validation: PASS"
    )

    print(
        "Eligible A/B/C medicines:",
        f"{len(dataframe):,}",
    )

    print(
        "Solver interface: appsi_highs"
    )

    print(
        "Pyomo:",
        pyomo.__version__,
    )

    print(
        "HiGHS:",
        highspy.Highs().version(),
    )

    allocations = []
    summaries = []

    for scenario, requirement_column in (
        SCENARIOS.items()
    ):

        print()
        print("-" * 112)

        print(
            "Validating scenario:",
            scenario,
        )

        print(
            "Budget level: 100% of "
            "whole-unit eligible B100"
        )

        allocation, summary = (
            solve_full_funding(
                dataframe=dataframe,
                scenario=scenario,
                requirement_column=(
                    requirement_column
                ),
            )
        )

        allocations.append(
            allocation
        )

        summaries.append(
            summary
        )

        print(
            "  Theta A:",
            f"{summary['Theta_A']:.8f}",
        )

        print(
            "  Theta B:",
            f"{summary['Theta_B']:.8f}",
        )

        print(
            "  Theta C:",
            f"{summary['Theta_C']:.8f}",
        )

        print(
            "  Required units:",
            f"{summary['Total_Required_Units']:,}",
        )

        print(
            "  Allocated units:",
            f"{summary['Total_Allocated_Units']:,}",
        )

        print(
            "  B100:",
            f"{summary['B100']:,.4f}",
        )

        print(
            "  Allocated value:",
            f"{summary['Total_Allocated_Value']:,.4f}",
        )

        print(
            "  Allocation mismatches:",
            summary[
                "Full_Allocation_Mismatch_Count"
            ],
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

    validate_results(
        summary_df
    )

    print()
    print(
        "FULL-FUNDING MATHEMATICAL "
        "VALIDATION: PASS"
    )

    governance = (
        build_governance(
            dataframe
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    allocation_df.to_parquet(
        OUTPUT_DIR
        / "milp_full_funding_allocations.parquet",
        index=False,
        engine="pyarrow",
    )

    allocation_df.to_csv(
        OUTPUT_DIR
        / "milp_full_funding_allocations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "milp_full_funding_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    by_abc.to_csv(
        OUTPUT_DIR
        / "milp_full_funding_by_abc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "milp_solver_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        allocations=allocation_df,
        summary=summary_df,
        by_abc=by_abc,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("-" * 112)
    print("FULL-FUNDING VALIDATION SUMMARY")
    print("-" * 112)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.6f}"
            ),
        )
    )

    print()
    print("-" * 112)
    print("FULL-FUNDING VALIDATION BY ABC")
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
    print("SOLVER GOVERNANCE")
    print("-" * 112)

    print(
        governance.to_string(
            index=False
        )
    )

    print()
    print("-" * 112)
    print("INTERPRETATION")
    print("-" * 112)

    print(
        "At B100, every eligible "
        "medicine must receive its full "
        "modeled net requirement."
    )

    print(
        "A/B/C minimum proportional "
        "coverage must equal 1.0."
    )

    print(
        "Passing this test validates the "
        "core MILP mechanics before "
        "budget-constrained scenarios "
        "are permitted."
    )

    unresolved_ab_count = int(
        governance[
            "High_Priority_AB_Unresolved_Count"
        ].iloc[0]
    )

    print(
        "The validation covers eligible "
        "A/B/C records only; it does not "
        f"resolve the {unresolved_ab_count:,} "
        "excluded high-priority A/B "
        "planning-input records."
    )

    print(
        "No 60%-90% budget optimization "
        "has been executed."
    )

    print()
    print("=" * 112)
    print(
        "100% FULL-FUNDING MILP "
        "VALIDATION COMPLETE"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
    