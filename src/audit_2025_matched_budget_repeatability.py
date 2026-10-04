from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

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
    / "retrospective_2025_matched_budget_repeatability_audit_v1"
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
    "FULL_REPEATABILITY_AUDIT_V1"
)

EXPECTED_ELIGIBLE = 2197

REFERENCE_FACTORS = [
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

VALUE_TOLERANCE = 0.05


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
# LOAD STORED STEP 30 OUTPUT
# ============================================================

def load_stored_allocations() -> pd.DataFrame:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        dataframe = connection.execute(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_allocations
            ORDER BY
                Reference_Budget_Percent,
                Requirement_Scenario,
                Medicine_Key
            """
        ).df()

    finally:

        connection.close()

    expected_rows = (
        EXPECTED_ELIGIBLE
        *
        len(
            matched.SCENARIOS
        )
        *
        len(
            REFERENCE_FACTORS
        )
    )

    if len(dataframe) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows:,} stored "
            f"allocation rows; found "
            f"{len(dataframe):,}"
        )

    return dataframe


# ============================================================
# ONE STORED CASE
# ============================================================

def get_stored_case(
    stored: pd.DataFrame,
    scenario: str,
    reference_factor: float,
) -> pd.DataFrame:

    result = stored.loc[
        stored[
            "Requirement_Scenario"
        ].eq(
            scenario
        )
        &
        np.isclose(
            stored[
                "Reference_Budget_Factor"
            ].astype(float),
            reference_factor,
        )
    ].copy()

    if len(result) != EXPECTED_ELIGIBLE:

        raise ValueError(
            f"Stored case {scenario} "
            f"@ {reference_factor:.0%} "
            f"contains {len(result):,} rows."
        )

    return result


# ============================================================
# COMPARE CASES
# ============================================================

def compare_case(
    stored_case: pd.DataFrame,
    rerun_case: pd.DataFrame,
    scenario: str,
    reference_factor: float,
    matched_budget: float,
    scenario_factor: float,
) -> dict:

    left = (
        stored_case[
            [
                "Medicine_Key",
                "Allocated_Units",
                "Allocated_Value",
                "Simulated_Annual_Issuance_Shortfall_Units",
                "Simulated_Residual_Inventory_Units",
            ]
        ]
        .sort_values(
            "Medicine_Key"
        )
        .reset_index(
            drop=True
        )
    )

    right = (
        rerun_case[
            [
                "Medicine_Key",
                "Allocated_Units",
                "Allocated_Value",
                "Simulated_Annual_Issuance_Shortfall_Units",
                "Simulated_Residual_Inventory_Units",
            ]
        ]
        .sort_values(
            "Medicine_Key"
        )
        .reset_index(
            drop=True
        )
    )

    if not left[
        "Medicine_Key"
    ].equals(
        right[
            "Medicine_Key"
        ]
    ):

        raise RuntimeError(
            "Medicine_Key alignment failed."
        )

    mismatch_count = int(
        (
            left[
                "Allocated_Units"
            ].to_numpy()
            !=
            right[
                "Allocated_Units"
            ].to_numpy()
        )
        .sum()
    )

    stored_hash = allocation_hash(
        stored_case
    )

    rerun_hash = allocation_hash(
        rerun_case
    )

    stored_value = float(
        left[
            "Allocated_Value"
        ].sum()
    )

    rerun_value = float(
        right[
            "Allocated_Value"
        ].sum()
    )

    stored_shortfall = int(
        left[
            "Simulated_Annual_Issuance_Shortfall_Units"
        ].sum()
    )

    rerun_shortfall = int(
        right[
            "Simulated_Annual_Issuance_Shortfall_Units"
        ].sum()
    )

    stored_residual = int(
        left[
            "Simulated_Residual_Inventory_Units"
        ].sum()
    )

    rerun_residual = int(
        right[
            "Simulated_Residual_Inventory_Units"
        ].sum()
    )

    exact_allocation_pass = bool(
        mismatch_count == 0
        and
        stored_hash == rerun_hash
    )

    outcome_pass = bool(
        stored_shortfall
        ==
        rerun_shortfall
        and
        stored_residual
        ==
        rerun_residual
        and
        abs(
            stored_value
            -
            rerun_value
        )
        <=
        VALUE_TOLERANCE
    )

    return {
        "Requirement_Scenario":
            scenario,

        "Reference_Budget_Percent":
            reference_factor * 100,

        "Matched_Peso_Budget":
            matched_budget,

        "Scenario_Specific_Budget_Percent":
            scenario_factor * 100,

        "Allocation_Mismatch_Count":
            mismatch_count,

        "Stored_SHA256":
            stored_hash,

        "Rerun_SHA256":
            rerun_hash,

        "Stored_Allocated_Value":
            stored_value,

        "Rerun_Allocated_Value":
            rerun_value,

        "Allocated_Value_Difference":
            rerun_value
            -
            stored_value,

        "Stored_Simulated_Shortfall_Units":
            stored_shortfall,

        "Rerun_Simulated_Shortfall_Units":
            rerun_shortfall,

        "Shortfall_Unit_Difference":
            rerun_shortfall
            -
            stored_shortfall,

        "Stored_Simulated_Residual_Units":
            stored_residual,

        "Rerun_Simulated_Residual_Units":
            rerun_residual,

        "Residual_Unit_Difference":
            rerun_residual
            -
            stored_residual,

        "Exact_Allocation_REPEAT_PASS":
            exact_allocation_pass,

        "Outcome_REPEAT_PASS":
            outcome_pass,

        "Case_REPEAT_PASS":
            bool(
                exact_allocation_pass
                and
                outcome_pass
            ),

        "Technical_Stage7_Run":
            False,

        "Solver_Configuration":
            (
                "parallel=off; "
                "threads=1; "
                "random_seed=0"
            ),

        "Audit_Version":
            AUDIT_VERSION,
    }


# ============================================================
# DATABASE
# ============================================================

def write_database(
    audit: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "audit_df",
            audit,
        )

        connection.register(
            "gov_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_full_repeatability_audit AS
            SELECT *
            FROM audit_df
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            retrospective_2025_matched_budget_full_repeatability_governance AS
            SELECT *
            FROM gov_df
            """
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 122)
    print("CDH INVENTORY DSS")
    print("2025 MATCHED-BUDGET FULL 15-SCENARIO REPEATABILITY AUDIT")
    print("=" * 122)
    print()

    full_data = (
        retro.load_inputs()
    )

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

    stored = (
        load_stored_allocations()
    )

    b100 = (
        matched.load_b100()
    )

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
        "Stored Step 30 allocation rows:",
        f"{len(stored):,}",
    )

    print(
        "Cases to independently rerun:",
        15,
    )

    print(
        "Technical Medicine_Key "
        "tie-break: OFF"
    )

    print(
        "Deterministic HiGHS:",
        "parallel=off; threads=1; random_seed=0",
    )

    print()

    rows = []

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

            scenario_factor = (
                matched_budget
                /
                b100[
                    scenario
                ]
            )

            print("-" * 122)

            print(
                "Repeat audit:",
                scenario,
                "@",
                f"{reference_factor:.0%} "
                "POINT-reference budget",
            )

            rerun, rerun_summary = (
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
                rerun,
                rerun_summary,
            )

            stored_case = (
                get_stored_case(
                    stored=stored,
                    scenario=scenario,
                    reference_factor=(
                        reference_factor
                    ),
                )
            )

            result = compare_case(
                stored_case=stored_case,
                rerun_case=rerun,
                scenario=scenario,
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

            rows.append(
                result
            )

            print(
                "  Allocation mismatches:",
                result[
                    "Allocation_Mismatch_Count"
                ],
            )

            print(
                "  Hash identical:",
                (
                    result[
                        "Stored_SHA256"
                    ]
                    ==
                    result[
                        "Rerun_SHA256"
                    ]
                ),
            )

            print(
                "  Shortfall unit difference:",
                result[
                    "Shortfall_Unit_Difference"
                ],
            )

            print(
                "  Residual unit difference:",
                result[
                    "Residual_Unit_Difference"
                ],
            )

            print(
                "  Case repeatability:",
                (
                    "PASS"
                    if result[
                        "Case_REPEAT_PASS"
                    ]
                    else "FAIL"
                ),
            )

    audit = pd.DataFrame(
        rows
    )

    if len(audit) != 15:

        raise RuntimeError(
            "Expected 15 audit rows."
        )

    if not audit[
        "Case_REPEAT_PASS"
    ].all():

        failed = audit.loc[
            ~audit[
                "Case_REPEAT_PASS"
            ],
            [
                "Requirement_Scenario",
                "Reference_Budget_Percent",
                "Allocation_Mismatch_Count",
                "Shortfall_Unit_Difference",
                "Residual_Unit_Difference",
            ],
        ]

        raise RuntimeError(
            "FULL REPEATABILITY AUDIT "
            "FAILED:\n"
            f"{failed.to_string(index=False)}"
        )

    governance = pd.DataFrame(
        [
            {
                "Audit_Version":
                    AUDIT_VERSION,

                "Target_Year":
                    2025,

                "Planning_Origin":
                    "END_OF_2024",

                "Cases_Repeated":
                    15,

                "Eligible_Medicine_Count":
                    EXPECTED_ELIGIBLE,

                "Total_Decisions_Checked":
                    (
                        EXPECTED_ELIGIBLE
                        *
                        15
                    ),

                "Technical_Stage7_Run":
                    False,

                "Solver_Parallel":
                    "off",

                "Solver_Threads":
                    1,

                "Solver_Random_Seed":
                    0,

                "Exact_Allocation_Repeatability_All_Cases":
                    True,

                "Outcome_Repeatability_All_Cases":
                    True,

                "Actual_2025_Entered_Optimization":
                    False,

                "Audit_Status":
                    "PASS",
            }
        ]
    )

    audit.to_csv(
        OUTPUT_DIR
        / "full_15_case_repeatability_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / "full_15_case_repeatability_governance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_database(
        audit,
        governance,
    )

    print()
    print("=" * 122)
    print("FULL 15-CASE REPEATABILITY AUDIT")
    print("=" * 122)

    print(
        audit[
            [
                "Requirement_Scenario",
                "Reference_Budget_Percent",
                "Allocation_Mismatch_Count",
                "Allocated_Value_Difference",
                "Shortfall_Unit_Difference",
                "Residual_Unit_Difference",
                "Case_REPEAT_PASS",
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
                f"{x:,.6f}"
            ),
        )
    )

    print()
    print(
        "Exact allocation repeatability "
        "across all 15 cases: PASS"
    )

    print(
        "Total medicine-level allocation "
        "decisions checked:",
        f"{EXPECTED_ELIGIBLE * 15:,}",
    )

    print()
    print(
        "STEP 30R1 COMPLETE"
    )


if __name__ == "__main__":
    main()
