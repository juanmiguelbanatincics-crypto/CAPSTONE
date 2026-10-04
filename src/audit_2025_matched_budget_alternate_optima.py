from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pyomo.environ import (
    Binary,
    Constraint,
    Expression,
    NonNegativeIntegers,
    Objective,
    Param,
    Set,
    Var,
    maximize,
    minimize,
    value,
)

import run_2025_retrospective_simulation as retro
import run_2025_matched_budget_comparison as matched


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
    / "retrospective_2025_matched_budget_altopt_audit_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GOVERNANCE
# ============================================================

AUDIT_VERSION = (
    "RETROSPECTIVE_2025_MATCHED_BUDGET_"
    "ALTERNATE_OPTIMUM_AUDIT_V1"
)

REFERENCE_FACTORS = [
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

EXPECTED_ELIGIBLE = 2197

VALUE_CHECK_TOLERANCE = 0.10

ABC_CLASSES = [
    "A",
    "B",
    "C",
]


# ============================================================
# HASH
# ============================================================

def allocation_hash(
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
        .reset_index(
            drop=True
        )
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


# ============================================================
# LOAD STORED STEP 30
# ============================================================

def load_step30():

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        allocations = connection.execute(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_allocations
            ORDER BY
                Reference_Budget_Percent,
                Requirement_Scenario,
                Medicine_Key
            """
        ).df()

        summary = connection.execute(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_summary
            ORDER BY
                Reference_Budget_Percent,
                Requirement_Scenario
            """
        ).df()

    finally:

        connection.close()

    return (
        allocations,
        summary,
    )


# ============================================================
# GET STORED CASE
# ============================================================

def get_stored_case(
    allocations: pd.DataFrame,
    scenario: str,
    reference_factor: float,
) -> pd.DataFrame:

    result = allocations.loc[
        allocations[
            "Requirement_Scenario"
        ].eq(
            scenario
        )
        &
        np.isclose(
            allocations[
                "Reference_Budget_Factor"
            ].astype(float),
            reference_factor,
        )
    ].copy()

    if len(result) != EXPECTED_ELIGIBLE:

        raise ValueError(
            f"Stored case {scenario} "
            f"@ {reference_factor:.0%} "
            f"has {len(result):,} rows."
        )

    return result


def get_stored_summary(
    summary: pd.DataFrame,
    scenario: str,
    reference_factor: float,
) -> pd.Series:

    result = summary.loc[
        summary[
            "Requirement_Scenario"
        ].eq(
            scenario
        )
        &
        np.isclose(
            summary[
                "Reference_Budget_Percent"
            ].astype(float),
            reference_factor
            *
            100,
        )
    ]

    if len(result) != 1:

        raise ValueError(
            f"Expected one stored summary "
            f"for {scenario} "
            f"@ {reference_factor:.0%}."
        )

    return result.iloc[0]


# ============================================================
# BUILD STAGE-6-OPTIMAL FEASIBLE REGION
# ============================================================

def build_altopt_model(
    eligible: pd.DataFrame,
    requirement_column: str,
    scenario_factor: float,
    stage_summary: dict,
):

    model_columns = (
        retro.MODEL_BASE_COLUMNS
        +
        [
            requirement_column,
        ]
    )

    model_data = eligible[
        model_columns
    ].copy()

    model, b100, budget = (
        retro.build_model(
            model_data=model_data,
            requirement_column=(
                requirement_column
            ),
            budget_factor=(
                scenario_factor
            ),
        )
    )

    # --------------------------------------------------------
    # REMOVE NORMAL OBJECTIVES
    # --------------------------------------------------------

    model.obj_theta_A.deactivate()
    model.obj_value_A.deactivate()
    model.obj_theta_B.deactivate()
    model.obj_value_B.deactivate()
    model.obj_theta_C.deactivate()
    model.obj_value_C.deactivate()
    model.obj_tie.deactivate()

    # --------------------------------------------------------
    # FIX THE SIX SUBSTANTIVE OBJECTIVE LEVELS
    #
    # These reproduce the same preservation tolerances used
    # by the production retrospective optimizer.
    # --------------------------------------------------------

    model.audit_keep_theta_A = Constraint(
        expr=(
            model.theta_A
            >=
            max(
                0.0,
                float(
                    stage_summary[
                        "Theta_A_Star"
                    ]
                )
                -
                retro.THETA_TOLERANCE,
            )
        )
    )

    model.audit_keep_value_A = Constraint(
        expr=(
            model.value_A
            >=
            float(
                stage_summary[
                    "A_Value_Star"
                ]
            )
            -
            retro.VALUE_TOLERANCE
        )
    )

    model.audit_keep_theta_B = Constraint(
        expr=(
            model.theta_B
            >=
            max(
                0.0,
                float(
                    stage_summary[
                        "Theta_B_Star"
                    ]
                )
                -
                retro.THETA_TOLERANCE,
            )
        )
    )

    model.audit_keep_value_B = Constraint(
        expr=(
            model.value_B
            >=
            float(
                stage_summary[
                    "B_Value_Star"
                ]
            )
            -
            retro.VALUE_TOLERANCE
        )
    )

    model.audit_keep_theta_C = Constraint(
        expr=(
            model.theta_C
            >=
            max(
                0.0,
                float(
                    stage_summary[
                        "Theta_C_Star"
                    ]
                )
                -
                retro.THETA_TOLERANCE,
            )
        )
    )

    model.audit_keep_value_C = Constraint(
        expr=(
            model.value_C
            >=
            float(
                stage_summary[
                    "C_Value_Star"
                ]
            )
            -
            retro.VALUE_TOLERANCE
        )
    )

    # ========================================================
    # POST-HOC ACTUAL OUTCOME INFORMATION
    #
    # This enters ONLY AFTER all six planning objective
    # levels above are fixed.
    # ========================================================

    actual = dict(
        zip(
            eligible[
                "Medicine_Key"
            ],
            eligible[
                "Actual_2025_Annual_Issuance"
            ]
            .astype(int),
        )
    )

    stock = dict(
        zip(
            eligible[
                "Medicine_Key"
            ],
            np.round(
                eligible[
                    "Planning_Stock_Qty"
                ]
                .astype(float)
            )
            .astype(int),
        )
    )

    deficit = {
        key:
            max(
                0,
                actual[key]
                -
                stock[key],
            )
        for key in actual
    }

    positive_deficit_keys = [
        key
        for key, qty in deficit.items()
        if qty > 0
    ]

    requirement = dict(
        zip(
            eligible[
                "Medicine_Key"
            ],
            eligible[
                requirement_column
            ]
            .astype(int),
        )
    )

    big_m = {
        key:
            max(
                1,
                deficit[key],
                requirement[key],
            )
        for key in positive_deficit_keys
    }

    model.OUTCOME_POS = Set(
        initialize=(
            positive_deficit_keys
        ),
        ordered=True,
    )

    model.Actual_Deficit = Param(
        model.OUTCOME_POS,
        initialize={
            key:
                deficit[key]
            for key
            in positive_deficit_keys
        },
        within=NonNegativeIntegers,
    )

    model.Outcome_M = Param(
        model.OUTCOME_POS,
        initialize=big_m,
        within=NonNegativeIntegers,
    )

    # Exact:
    #
    # short_i = max(0, actual_i - stock_i - x_i)
    #
    model.shortfall = Var(
        model.OUTCOME_POS,
        domain=NonNegativeIntegers,
        bounds=lambda m, i: (
            0,
            int(
                value(
                    m.Actual_Deficit[i]
                )
            ),
        ),
    )

    model.shortfall_positive = Var(
        model.OUTCOME_POS,
        domain=Binary,
    )

    # --------------------------------------------------------
    # BRANCH LINEARIZATION
    # --------------------------------------------------------

    def sign_upper_rule(
        m,
        i,
    ):

        return (
            m.Actual_Deficit[i]
            -
            m.x[i]
            <=
            m.Outcome_M[i]
            *
            m.shortfall_positive[i]
        )

    def sign_lower_rule(
        m,
        i,
    ):

        return (
            m.Actual_Deficit[i]
            -
            m.x[i]
            >=
            -
            m.Outcome_M[i]
            *
            (
                1
                -
                m.shortfall_positive[i]
            )
        )

    def shortfall_lower_rule(
        m,
        i,
    ):

        return (
            m.shortfall[i]
            >=
            m.Actual_Deficit[i]
            -
            m.x[i]
        )

    def shortfall_upper_positive_rule(
        m,
        i,
    ):

        return (
            m.shortfall[i]
            <=
            m.Actual_Deficit[i]
            -
            m.x[i]
            +
            m.Outcome_M[i]
            *
            (
                1
                -
                m.shortfall_positive[i]
            )
        )

    def shortfall_upper_zero_rule(
        m,
        i,
    ):

        return (
            m.shortfall[i]
            <=
            m.Outcome_M[i]
            *
            m.shortfall_positive[i]
        )

    model.outcome_sign_upper = Constraint(
        model.OUTCOME_POS,
        rule=sign_upper_rule,
    )

    model.outcome_sign_lower = Constraint(
        model.OUTCOME_POS,
        rule=sign_lower_rule,
    )

    model.outcome_short_lower = Constraint(
        model.OUTCOME_POS,
        rule=shortfall_lower_rule,
    )

    model.outcome_short_upper_positive = Constraint(
        model.OUTCOME_POS,
        rule=(
            shortfall_upper_positive_rule
        ),
    )

    model.outcome_short_upper_zero = Constraint(
        model.OUTCOME_POS,
        rule=shortfall_upper_zero_rule,
    )

    # --------------------------------------------------------
    # SHORTFALL ECONOMIC VALUE
    # --------------------------------------------------------

    model.posthoc_shortfall_value = Expression(
        expr=sum(
            model.C[i]
            *
            model.shortfall[i]
            for i in model.OUTCOME_POS
        )
    )

    model.obj_min_posthoc_shortfall = Objective(
        expr=model.posthoc_shortfall_value,
        sense=minimize,
    )

    model.obj_max_posthoc_shortfall = Objective(
        expr=model.posthoc_shortfall_value,
        sense=maximize,
    )

    model.obj_max_posthoc_shortfall.deactivate()

    return (
        model,
        b100,
        budget,
    )


# ============================================================
# EXTRACT EXTREME SOLUTION
# ============================================================

def extract_extreme(
    model,
    eligible: pd.DataFrame,
    requirement_column: str,
    scenario: str,
    reference_factor: float,
    variant: str,
    baseline: pd.DataFrame,
    stage_summary: dict,
):

    result = eligible[
        [
            "Medicine_Key",
            "Brand_Name",
            "Generic_Name",
            "Unit",
            "ABC_AsOf",
            "Planning_Cost",
            "Planning_Stock_Qty",
            "Actual_2025_Annual_Issuance",
        ]
    ].copy()

    result[
        "Net_Requirement"
    ] = (
        eligible[
            requirement_column
        ]
        .astype(int)
        .to_numpy()
    )

    result[
        "Allocated_Units"
    ] = [
        int(
            round(
                value(
                    model.x[key]
                )
            )
        )
        for key in eligible[
            "Medicine_Key"
        ]
    ]

    result[
        "Baseline_Stock_Units"
    ] = (
        np.round(
            result[
                "Planning_Stock_Qty"
            ]
        )
        .astype(int)
    )

    result[
        "Simulated_Available_Units"
    ] = (
        result[
            "Baseline_Stock_Units"
        ]
        +
        result[
            "Allocated_Units"
        ]
    )

    result[
        "Simulated_Shortfall_Units"
    ] = np.maximum(
        0,
        (
            result[
                "Actual_2025_Annual_Issuance"
            ]
            -
            result[
                "Simulated_Available_Units"
            ]
        ),
    )

    result[
        "Simulated_Residual_Units"
    ] = np.maximum(
        0,
        (
            result[
                "Simulated_Available_Units"
            ]
            -
            result[
                "Actual_2025_Annual_Issuance"
            ]
        ),
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
        "Shortfall_Value"
    ] = (
        result[
            "Simulated_Shortfall_Units"
        ]
        *
        result[
            "Planning_Cost"
        ]
    )

    result[
        "Residual_Value"
    ] = (
        result[
            "Simulated_Residual_Units"
        ]
        *
        result[
            "Planning_Cost"
        ]
    )

    result[
        "Requirement_Scenario"
    ] = scenario

    result[
        "Reference_Budget_Percent"
    ] = (
        reference_factor
        *
        100
    )

    result[
        "Extreme_Variant"
    ] = variant

    result[
        "Audit_Version"
    ] = AUDIT_VERSION

    # --------------------------------------------------------
    # CHECK OBJECTIVE REPRESENTATION
    # --------------------------------------------------------

    computed_shortfall_value = float(
        result[
            "Shortfall_Value"
        ].sum()
    )

    model_shortfall_value = float(
        value(
            model.posthoc_shortfall_value
        )
    )

    if abs(
        computed_shortfall_value
        -
        model_shortfall_value
    ) > VALUE_CHECK_TOLERANCE:

        raise RuntimeError(
            "Post-hoc shortfall objective "
            "does not reproduce computed "
            "shortfall value."
        )

    # --------------------------------------------------------
    # CHECK STAGE-6 OBJECTIVE PRESERVATION
    # --------------------------------------------------------

    for abc_class in ABC_CLASSES:

        group = result.loc[
            result[
                "ABC_AsOf"
            ].eq(
                abc_class
            )
        ]

        class_value = float(
            group[
                "Allocated_Value"
            ].sum()
        )

        positive = group.loc[
            group[
                "Net_Requirement"
            ].gt(0)
        ]

        if positive.empty:

            observed_min = 1.0

        else:

            observed_min = float(
                (
                    positive[
                        "Allocated_Units"
                    ]
                    /
                    positive[
                        "Net_Requirement"
                    ]
                )
                .min()
            )

        theta_key = (
            f"Theta_{abc_class}_Star"
        )

        value_key = (
            f"{abc_class}_Value_Star"
        )

        if (
            observed_min
            +
            retro.COVERAGE_TOLERANCE
            <
            float(
                stage_summary[
                    theta_key
                ]
            )
            -
            retro.THETA_TOLERANCE
        ):

            raise RuntimeError(
                f"{abc_class} coverage "
                "objective not preserved."
            )

        if (
            class_value
            <
            float(
                stage_summary[
                    value_key
                ]
            )
            -
            retro.VALUE_TOLERANCE
            -
            1e-6
        ):

            raise RuntimeError(
                f"{abc_class} value "
                "objective not preserved."
            )

    # --------------------------------------------------------
    # BASELINE ALLOCATION DIFFERENCE
    # --------------------------------------------------------

    baseline_compare = (
        baseline[
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

    extreme_compare = (
        result[
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
            baseline_compare[
                "Allocated_Units"
            ]
            .to_numpy()
            !=
            extreme_compare[
                "Allocated_Units"
            ]
            .to_numpy()
        )
        .sum()
    )

    return (
        result,
        {
            "Requirement_Scenario":
                scenario,

            "Reference_Budget_Percent":
                reference_factor
                *
                100,

            "Extreme_Variant":
                variant,

            "Allocated_Value":
                float(
                    result[
                        "Allocated_Value"
                    ].sum()
                ),

            "Shortfall_Units":
                int(
                    result[
                        "Simulated_Shortfall_Units"
                    ].sum()
                ),

            "Shortfall_Value":
                computed_shortfall_value,

            "Residual_Units":
                int(
                    result[
                        "Simulated_Residual_Units"
                    ].sum()
                ),

            "Residual_Value":
                float(
                    result[
                        "Residual_Value"
                    ].sum()
                ),

            "Allocation_Difference_From_"
            "Step30_Count":
                mismatch_count,

            "Allocation_SHA256":
                allocation_hash(
                    result
                ),

            "Audit_Version":
                AUDIT_VERSION,
        },
    )


# ============================================================
# ECONOMIC IDENTITY CHECK
# ============================================================

def validate_value_identity(
    dataframe: pd.DataFrame,
) -> float:

    actual_value = float(
        (
            dataframe[
                "Actual_2025_Annual_Issuance"
            ]
            *
            dataframe[
                "Planning_Cost"
            ]
        )
        .sum()
    )

    baseline_stock_value = float(
        (
            dataframe[
                "Baseline_Stock_Units"
            ]
            *
            dataframe[
                "Planning_Cost"
            ]
        )
        .sum()
    )

    allocated_value = float(
        dataframe[
            "Allocated_Value"
        ].sum()
    )

    shortfall_value = float(
        dataframe[
            "Shortfall_Value"
        ].sum()
    )

    residual_value = float(
        dataframe[
            "Residual_Value"
        ].sum()
    )

    difference = (
        (
            residual_value
            -
            shortfall_value
        )
        -
        (
            baseline_stock_value
            +
            allocated_value
            -
            actual_value
        )
    )

    if abs(
        difference
    ) > VALUE_CHECK_TOLERANCE:

        raise RuntimeError(
            "Residual-shortfall economic "
            "identity failed."
        )

    return difference


# ============================================================
# POLICY ROBUSTNESS
# ============================================================

def build_policy_robustness(
    ranges: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for reference_budget, group in (
        ranges.groupby(
            "Reference_Budget_Percent",
            sort=True,
        )
    ):

        baseline_sorted = (
            group.sort_values(
                "Baseline_Shortfall_Value"
            )
        )

        baseline_best = str(
            baseline_sorted.iloc[0][
                "Requirement_Scenario"
            ]
        )

        robust_candidates = []

        for _, candidate in (
            group.iterrows()
        ):

            others = group.loc[
                ~group[
                    "Requirement_Scenario"
                ]
                .eq(
                    candidate[
                        "Requirement_Scenario"
                    ]
                )
            ]

            candidate_worst = float(
                candidate[
                    "Maximum_Stage6_Optimal_"
                    "Shortfall_Value"
                ]
            )

            competitor_best = float(
                others[
                    "Minimum_Stage6_Optimal_"
                    "Shortfall_Value"
                ]
                .min()
            )

            if (
                candidate_worst
                <
                competitor_best
                -
                VALUE_CHECK_TOLERANCE
            ):

                robust_candidates.append(
                    (
                        str(
                            candidate[
                                "Requirement_Scenario"
                            ]
                        ),
                        competitor_best
                        -
                        candidate_worst,
                    )
                )

        if len(
            robust_candidates
        ) == 1:

            robust_policy = (
                robust_candidates[0][0]
            )

            separation = float(
                robust_candidates[0][1]
            )

        else:

            robust_policy = (
                "NO_SINGLE_ROBUST_WINNER"
            )

            separation = np.nan

        rows.append(
            {
                "Reference_Budget_Percent":
                    reference_budget,

                "Baseline_Best_Policy":
                    baseline_best,

                "Robust_Best_Policy":
                    robust_policy,

                "Baseline_Winner_Robust_PASS":
                    bool(
                        robust_policy
                        ==
                        baseline_best
                    ),

                "Worst_Case_Separation_From_"
                "Best_Competitor":
                    separation,

                "Interpretation":
                    (
                        "ROBUST_TO_STAGE6_"
                        "ALTERNATE_OPTIMA"
                        if robust_policy
                        ==
                        baseline_best
                        else
                        "POLICY_RANKING_"
                        "SENSITIVE_OR_OVERLAPPING"
                    ),

                "Audit_Version":
                    AUDIT_VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DATABASE
# ============================================================

def write_database(
    extreme_allocations,
    extreme_summary,
    ranges,
    robustness,
    governance,
):

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "alloc_df",
            extreme_allocations,
        )

        connection.register(
            "extreme_df",
            extreme_summary,
        )

        connection.register(
            "range_df",
            ranges,
        )

        connection.register(
            "robust_df",
            robustness,
        )

        connection.register(
            "gov_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_altopt_allocations AS
            SELECT * FROM alloc_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_altopt_extremes AS
            SELECT * FROM extreme_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_altopt_ranges AS
            SELECT * FROM range_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_altopt_policy_robustness AS
            SELECT * FROM robust_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_altopt_governance AS
            SELECT * FROM gov_df
            """
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 124)
    print("CDH INVENTORY DSS")
    print("2025 MATCHED-BUDGET ALTERNATE-OPTIMUM OUTCOME SENSITIVITY AUDIT")
    print("=" * 124)
    print()

    full_data = retro.load_inputs()

    retro.validate_inputs(
        full_data
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

    (
        stored_allocations,
        stored_summary,
    ) = load_step30()

    b100 = matched.load_b100()

    reference_b100 = (
        b100[
            matched.REFERENCE_SCENARIO
        ]
    )

    print(
        "Temporal validation: PASS"
    )

    print(
        "Eligible medicines:",
        f"{len(eligible):,}",
    )

    print(
        "Matched-budget levels under audit:",
        ", ".join(f"{factor:.0%}" for factor in REFERENCE_FACTORS) + " of POINT B100",
    )

    print(
        "Policies:",
        ", ".join(
            matched.SCENARIOS.keys()
        ),
    )

    print(
        "Stage 7 technical tie-break:",
        "NOT USED",
    )

    print(
        "Actual 2025 outcome enters "
        "planning optimization:",
        "NO",
    )

    print(
        "Actual 2025 outcome enters "
        "post-hoc alternate-optimum audit:",
        "YES",
    )

    print()

    extreme_allocations = []
    extreme_rows = []
    range_rows = []

    for reference_factor in (
        REFERENCE_FACTORS
    ):

        matched_budget = (
            reference_b100
            *
            reference_factor
        )

        for scenario, requirement_column in (
            matched.SCENARIOS.items()
        ):

            print("-" * 124)

            print(
                "Alternate-optimum audit:",
                scenario,
                "@",
                f"{reference_factor:.0%} "
                "POINT-reference budget",
            )

            scenario_factor = (
                matched_budget
                /
                b100[
                    scenario
                ]
            )

            # ------------------------------------------------
            # RECONSTRUCT THE SIX OPTIMAL PLANNING LEVELS
            # ------------------------------------------------

            baseline_rerun, stage_summary = (
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
                baseline_rerun,
                stage_summary,
            )

            stored_case = get_stored_case(
                allocations=stored_allocations,
                scenario=scenario,
                reference_factor=(
                    reference_factor
                ),
            )

            if (
                allocation_hash(
                    baseline_rerun
                )
                !=
                allocation_hash(
                    stored_case
                )
            ):

                raise RuntimeError(
                    "Baseline reconstruction "
                    "does not match stored Step 30."
                )

            baseline_summary = (
                get_stored_summary(
                    summary=stored_summary,
                    scenario=scenario,
                    reference_factor=(
                        reference_factor
                    ),
                )
            )

            # ------------------------------------------------
            # BUILD ALTERNATE-OPTIMUM FEASIBLE REGION
            # ------------------------------------------------

            (
                model,
                model_b100,
                model_budget,
            ) = build_altopt_model(
                eligible=eligible,
                requirement_column=(
                    requirement_column
                ),
                scenario_factor=(
                    scenario_factor
                ),
                stage_summary=(
                    stage_summary
                ),
            )

            if abs(
                model_budget
                -
                matched_budget
            ) > VALUE_CHECK_TOLERANCE:

                raise RuntimeError(
                    "Matched budget reproduction failed."
                )

            solver = retro.make_solver(
                deterministic=True
            )

            solver.highs_options[
                "mip_rel_gap"
            ] = 0.0

            solver.highs_options[
                "mip_abs_gap"
            ] = 1e-6

            # =================================================
            # BEST POSSIBLE RETROSPECTIVE SHORTFALL
            # AMONG STAGE-6-OPTIMAL ALLOCATIONS
            # =================================================

            retro.solve_optimal(
                solver,
                model,
                (
                    f"{scenario} "
                    f"{reference_factor:.0%} "
                    "MIN post-hoc shortfall"
                ),
            )

            (
                min_allocation,
                min_summary,
            ) = extract_extreme(
                model=model,
                eligible=eligible,
                requirement_column=(
                    requirement_column
                ),
                scenario=scenario,
                reference_factor=(
                    reference_factor
                ),
                variant=(
                    "MIN_SHORTFALL_VALUE"
                ),
                baseline=stored_case,
                stage_summary=(
                    stage_summary
                ),
            )

            identity_min = (
                validate_value_identity(
                    min_allocation
                )
            )

            min_summary[
                "Economic_Identity_Difference"
            ] = identity_min

            extreme_allocations.append(
                min_allocation
            )

            extreme_rows.append(
                min_summary
            )

            # =================================================
            # WORST POSSIBLE RETROSPECTIVE SHORTFALL
            # AMONG SAME STAGE-6-OPTIMAL ALLOCATIONS
            # =================================================

            model.obj_min_posthoc_shortfall.deactivate()
            model.obj_max_posthoc_shortfall.activate()

            retro.solve_optimal(
                solver,
                model,
                (
                    f"{scenario} "
                    f"{reference_factor:.0%} "
                    "MAX post-hoc shortfall"
                ),
            )

            (
                max_allocation,
                max_summary,
            ) = extract_extreme(
                model=model,
                eligible=eligible,
                requirement_column=(
                    requirement_column
                ),
                scenario=scenario,
                reference_factor=(
                    reference_factor
                ),
                variant=(
                    "MAX_SHORTFALL_VALUE"
                ),
                baseline=stored_case,
                stage_summary=(
                    stage_summary
                ),
            )

            identity_max = (
                validate_value_identity(
                    max_allocation
                )
            )

            max_summary[
                "Economic_Identity_Difference"
            ] = identity_max

            extreme_allocations.append(
                max_allocation
            )

            extreme_rows.append(
                max_summary
            )

            baseline_shortfall_value = float(
                baseline_summary[
                    "Simulated_Shortfall_Value_"
                    "At_Planning_Cost"
                ]
            )

            minimum_value = float(
                min_summary[
                    "Shortfall_Value"
                ]
            )

            maximum_value = float(
                max_summary[
                    "Shortfall_Value"
                ]
            )

            if (
                baseline_shortfall_value
                <
                minimum_value
                -
                VALUE_CHECK_TOLERANCE
                or
                baseline_shortfall_value
                >
                maximum_value
                +
                VALUE_CHECK_TOLERANCE
            ):

                raise RuntimeError(
                    "Stored Step 30 result "
                    "falls outside alternate-"
                    "optimum outcome range."
                )

            range_width = (
                maximum_value
                -
                minimum_value
            )

            range_rows.append(
                {
                    "Requirement_Scenario":
                        scenario,

                    "Reference_Budget_Percent":
                        reference_factor
                        *
                        100,

                    "Matched_Peso_Budget":
                        matched_budget,

                    "Scenario_Specific_Budget_Percent":
                        scenario_factor
                        *
                        100,

                    "Baseline_Shortfall_Value":
                        baseline_shortfall_value,

                    "Minimum_Stage6_Optimal_"
                    "Shortfall_Value":
                        minimum_value,

                    "Maximum_Stage6_Optimal_"
                    "Shortfall_Value":
                        maximum_value,

                    "Stage6_Optimal_"
                    "Shortfall_Value_Range":
                        range_width,

                    "Baseline_Within_Range_PASS":
                        True,

                    "Min_Extreme_Allocation_"
                    "Difference_Count":
                        int(
                            min_summary[
                                "Allocation_Difference_From_"
                                "Step30_Count"
                            ]
                        ),

                    "Max_Extreme_Allocation_"
                    "Difference_Count":
                        int(
                            max_summary[
                                "Allocation_Difference_From_"
                                "Step30_Count"
                            ]
                        ),

                    "Outcome_Invariant_Within_"
                    "Value_Tolerance":
                        bool(
                            range_width
                            <=
                            VALUE_CHECK_TOLERANCE
                        ),

                    "Audit_Version":
                        AUDIT_VERSION,
                }
            )

            print(
                "  Step 30 shortfall value:",
                f"{baseline_shortfall_value:,.2f}",
            )

            print(
                "  Best Stage-6-optimal "
                "shortfall value:",
                f"{minimum_value:,.2f}",
            )

            print(
                "  Worst Stage-6-optimal "
                "shortfall value:",
                f"{maximum_value:,.2f}",
            )

            print(
                "  Outcome range width:",
                f"{range_width:,.2f}",
            )

            print(
                "  MIN allocation differences:",
                min_summary[
                    "Allocation_Difference_From_"
                    "Step30_Count"
                ],
            )

            print(
                "  MAX allocation differences:",
                max_summary[
                    "Allocation_Difference_From_"
                    "Step30_Count"
                ],
            )

    extreme_allocations_df = (
        pd.concat(
            extreme_allocations,
            ignore_index=True,
        )
    )

    extreme_summary_df = pd.DataFrame(
        extreme_rows
    )

    ranges = pd.DataFrame(
        range_rows
    )

    if not ranges[
        "Baseline_Within_Range_PASS"
    ].all():

        raise RuntimeError(
            "Baseline range audit failed."
        )

    robustness = (
        build_policy_robustness(
            ranges
        )
    )

    governance = pd.DataFrame(
        [
            {
                "Audit_Version":
                    AUDIT_VERSION,

                "Planning_Origin":
                    "END_OF_2024",

                "Target_Year":
                    2025,

                "Cases_Audited":
                    len(ranges),

                "Reference_Budget_Levels":
                    ",".join(
                        f"{factor:.0%}"
                        for factor in REFERENCE_FACTORS
                    )
                    + " OF POINT B100",

                "Policies_Audited":
                    (
                        "POINT,P80_FLOORED_CEIL,"
                        "P90_FLOORED_CEIL"
                    ),

                "Planning_Objectives_Fixed_Before_"
                "Actual_Outcome_Used":
                    True,

                "Substantive_Planning_Stages_Fixed":
                    "STAGES_1_TO_6",

                "Technical_Stage7_Run":
                    False,

                "Actual_2025_Entered_Planning_"
                "Optimization":
                    False,

                "Actual_2025_Used_PostHoc_For_"
                "Sensitivity":
                    True,

                "Sensitivity_Metric":
                    (
                        "SIMULATED_SHORTFALL_VALUE_"
                        "AT_PLANNING_COST"
                    ),

                "Economic_Identity_Validated":
                    True,

                "Actual_Stockout_Claim":
                    False,

                "Actual_Waste_Claim":
                    False,
            }
        ]
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    extreme_allocations_df.to_parquet(
        OUTPUT_DIR
        / "alternate_optimum_extreme_allocations.parquet",
        index=False,
        engine="pyarrow",
    )

    extreme_summary_df.to_csv(
        OUTPUT_DIR
        / "alternate_optimum_extremes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ranges.to_csv(
        OUTPUT_DIR
        / "alternate_optimum_ranges.csv",
        index=False,
        encoding="utf-8-sig",
    )

    robustness.to_csv(
        OUTPUT_DIR
        / "alternate_optimum_policy_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "alternate_optimum_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        extreme_allocations=(
            extreme_allocations_df
        ),
        extreme_summary=(
            extreme_summary_df
        ),
        ranges=ranges,
        robustness=robustness,
        governance=governance,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 124)
    print("ALTERNATE-OPTIMUM OUTCOME RANGES")
    print("=" * 124)

    print(
        ranges[
            [
                "Requirement_Scenario",
                "Reference_Budget_Percent",
                "Baseline_Shortfall_Value",
                "Minimum_Stage6_Optimal_Shortfall_Value",
                "Maximum_Stage6_Optimal_Shortfall_Value",
                "Stage6_Optimal_Shortfall_Value_Range",
                "Min_Extreme_Allocation_Difference_Count",
                "Max_Extreme_Allocation_Difference_Count",
                "Outcome_Invariant_Within_Value_Tolerance",
            ]
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
    print("=" * 124)
    print("CROSS-POLICY ROBUSTNESS")
    print("=" * 124)

    print(
        robustness.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:,.4f}"
            ),
        )
    )

    print()
    print("-" * 124)
    print("INTERPRETATION")
    print("-" * 124)

    print(
        "Actual 2025 issuance was not used "
        "to determine the six procurement "
        "objective levels."
    )

    print(
        "Only after those six objective "
        "levels were fixed was actual 2025 "
        "issuance used to search for the "
        "best and worst retrospective "
        "shortfall outcomes among equally "
        "acceptable Stage-6 solutions."
    )

    print(
        "Therefore this is a post-hoc "
        "alternate-optimum sensitivity "
        "analysis, not outcome-informed "
        "procurement optimization."
    )

    print(
        "A baseline policy winner is called "
        "robust only when its worst "
        "Stage-6-optimal shortfall value "
        "remains lower than every competing "
        "policy's best Stage-6-optimal "
        "shortfall value."
    )

    print()
    print("=" * 124)
    print(
        "STEP 30R2 ALTERNATE-OPTIMUM "
        "AUDIT COMPLETE"
    )
    print("=" * 124)


if __name__ == "__main__":
    main()