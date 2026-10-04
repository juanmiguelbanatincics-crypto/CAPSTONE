from __future__ import annotations

import duckdb

try:
    from .runtime_context import (
        load_runtime_context_from_env,
    )
except ImportError:
    from runtime_context import (
        load_runtime_context_from_env,
    )


CONTRACT_VERSION = (
    "CURRENT_PRESENTATION_CONTRACT_V1"
)

RUNTIME_CONTEXT = (
    load_runtime_context_from_env()
)

DATABASE_PATH = (
    RUNTIME_CONTEXT.database_path
)

TARGET_YEAR = int(
    RUNTIME_CONTEXT.planning_year
)

LATEST_COMPLETED_YEAR = int(
    RUNTIME_CONTEXT.latest_completed_year
)

HISTORICAL_START_YEAR = int(
    RUNTIME_CONTEXT.historical_start_year
)


def table_exists(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> bool:

    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?
        """,
        [table_name],
    ).fetchone()[0]

    return bool(result)


def required_sources() -> dict[str, str]:

    year = TARGET_YEAR

    return {
        "forecast":
            f"forecast_{year}_governed_candidate",

        "forecast_segment":
            (
                f"forecast_{year}_governed_"
                "candidate_segment_summary"
            ),

        "forecast_governance":
            (
                f"forecast_{year}_governed_"
                "candidate_governance"
            ),

        "planning":
            f"planning_{year}_governed_inputs",

        "planning_governance":
            (
                f"planning_{year}_governed_"
                "governance"
            ),

        "review_queue":
            (
                f"planning_{year}_governed_"
                "review_queue"
            ),

        "milp_governance":
            (
                f"planning_{year}_governed_"
                "milp_governance"
            ),

        "strict_summary":
            (
                f"planning_{year}_governed_"
                "milp_strict_summary"
            ),

        "strict_by_abc":
            (
                f"planning_{year}_governed_"
                "milp_strict_by_abc"
            ),

        "strict_allocations":
            (
                f"planning_{year}_governed_"
                "milp_strict_allocations"
            ),

        "strict_priority_audit":
            (
                f"planning_{year}_governed_"
                "milp_strict_priority_audit"
            ),

        "strict_governance":
            (
                f"planning_{year}_governed_"
                "milp_strict_governance"
            ),

        "full_validation":
            (
                f"planning_{year}_governed_"
                "milp_full_funding_validation"
            ),

        "full_by_abc":
            (
                f"planning_{year}_governed_"
                "milp_full_funding_by_abc"
            ),

        "full_allocations":
            (
                f"planning_{year}_governed_"
                "milp_full_funding_allocations"
            ),

        "full_governance":
            (
                f"planning_{year}_governed_"
                "milp_full_funding_governance"
            ),
    }


def validate_sources(
    connection: duckdb.DuckDBPyConnection,
    sources: dict[str, str],
) -> None:

    missing = [
        table
        for table in sources.values()
        if not table_exists(
            connection,
            table,
        )
    ]

    if missing:
        formatted = "\n".join(
            f"- {name}"
            for name in missing
        )

        raise ValueError(
            "CURRENT presentation contract "
            "cannot be built because required "
            "governed source tables are missing:\n"
            f"{formatted}"
        )


def build_contract() -> None:

    if RUNTIME_CONTEXT.run_mode not in {
        "TEST",
        "STAGING",
    }:
        raise ValueError(
            "The CURRENT presentation contract "
            "must first be built and validated "
            "inside TEST or STAGING. "
            "Direct live publication is not "
            "allowed by this builder."
        )

    sources = required_sources()

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:
        validate_sources(
            connection,
            sources,
        )

        connection.execute(
            "BEGIN TRANSACTION"
        )

        # ----------------------------------------------------
        # SYSTEM CONTEXT
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            current_system_context AS

            SELECT
                ?::VARCHAR
                    AS Contract_Version,

                ?::BIGINT
                    AS Historical_Start_Year,

                ?::BIGINT
                    AS Latest_Completed_Year,

                ?::BIGINT
                    AS Planning_Year,

                ?::VARCHAR
                    AS Candidate_Run_ID,

                ?::VARCHAR
                    AS Build_Run_Mode,

                'VALIDATED_CANDIDATE'
                    ::VARCHAR
                    AS Current_State,

                NULL::VARCHAR
                    AS Published_Version,

                'CANDIDATE_NOT_PUBLISHED'
                    ::VARCHAR
                    AS Publication_Status,

                FALSE::BOOLEAN
                    AS Automatic_Publication
            """,
            [
                CONTRACT_VERSION,
                HISTORICAL_START_YEAR,
                LATEST_COMPLETED_YEAR,
                TARGET_YEAR,
                RUNTIME_CONTEXT.run_id,
                RUNTIME_CONTEXT.run_mode,
            ],
        )

        # ----------------------------------------------------
        # FORECAST / PLANNING
        # ----------------------------------------------------

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_forecast AS
            SELECT *
            FROM {sources["forecast"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_forecast_segment_summary AS

            SELECT
                Segment_Type,
                Segment,

                Rows
                    AS Medicine_Count,

                Point_Forecast_Total,

                Hurdle_Reference_Total,

                P50_Total,
                P80_Total,
                P90_Total,

                Mean_Occurrence_Probability

            FROM {sources["forecast_segment"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_forecast_governance AS
            SELECT *
            FROM {sources["forecast_governance"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_planning_inputs AS
            SELECT *
            FROM {sources["planning"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_planning_governance AS
            SELECT *
            FROM {sources["planning_governance"]}
            """
        )

        # ----------------------------------------------------
        # REVIEW QUEUE + SUMMARY
        # ----------------------------------------------------

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_review_queue AS
            SELECT *
            FROM {sources["review_queue"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_review_summary AS

            SELECT
                COUNT(*)::BIGINT
                    AS Review_Row_Count,

                SUM(
                    CASE
                        WHEN Needs_Stock_Review
                        THEN 1 ELSE 0
                    END
                )::BIGINT
                    AS Stock_Review_Count,

                SUM(
                    CASE
                        WHEN Needs_Cost_Review
                        THEN 1 ELSE 0
                    END
                )::BIGINT
                    AS Cost_Review_Count,

                SUM(
                    CASE
                        WHEN High_Priority_AB_Unresolved
                        THEN 1 ELSE 0
                    END
                )::BIGINT
                    AS High_Priority_AB_Blocker_Count,

                SUM(
                    CASE
                        WHEN Requirement_Point > 0
                        THEN 1 ELSE 0
                    END
                )::BIGINT
                    AS Positive_Point_Requirement_Review_Count,

                SUM(
                    CASE
                        WHEN Requirement_P80 > 0
                        THEN 1 ELSE 0
                    END
                )::BIGINT
                    AS Positive_P80_Requirement_Review_Count,

                SUM(
                    CASE
                        WHEN Requirement_P90 > 0
                        THEN 1 ELSE 0
                    END
                )::BIGINT
                    AS Positive_P90_Requirement_Review_Count,

                SUM(
                    COALESCE(
                        Requirement_Point,
                        0
                    )
                )
                    AS Review_Point_Requirement_Total,

                SUM(
                    COALESCE(
                        Requirement_P80,
                        0
                    )
                )
                    AS Review_P80_Requirement_Total,

                SUM(
                    COALESCE(
                        Requirement_P90,
                        0
                    )
                )
                    AS Review_P90_Requirement_Total,

                '{CONTRACT_VERSION}'
                    ::VARCHAR
                    AS Review_Version

            FROM {sources["review_queue"]}
            """
        )

        # ----------------------------------------------------
        # STRICT 60-90% OPTIMIZATION
        # ----------------------------------------------------

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_strict_summary AS
            SELECT *
            FROM {sources["strict_summary"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_strict_by_abc AS
            SELECT *
            FROM {sources["strict_by_abc"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_strict_allocations AS
            SELECT *
            FROM {sources["strict_allocations"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_strict_priority_audit AS
            SELECT *
            FROM {sources["strict_priority_audit"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_strict_governance AS
            SELECT *
            FROM {sources["strict_governance"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_governance AS
            SELECT *
            FROM {sources["milp_governance"]}
            """
        )

        # ----------------------------------------------------
        # B100 FULL-FUNDING VALIDATION
        # ----------------------------------------------------

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_full_funding_validation AS
            SELECT *
            FROM {sources["full_validation"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_full_funding_by_abc AS
            SELECT *
            FROM {sources["full_by_abc"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_full_funding_allocations AS
            SELECT *
            FROM {sources["full_allocations"]}
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_full_funding_governance AS
            SELECT *
            FROM {sources["full_governance"]}
            """
        )

        # ----------------------------------------------------
        # APP-FACING 60-100% SCENARIO SELECTOR
        # ----------------------------------------------------

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            current_milp_budget_scenarios AS

            SELECT
                Requirement_Scenario,
                Budget_Factor,
                Budget_Percent,

                B100
                    AS MILP_B100,

                Scenario_Budget,

                FALSE::BOOLEAN
                    AS Budget_Is_Actual_CDH_Budget,

                (
                    'Modeled percentage of '
                    'whole-unit eligible '
                    'full-funding requirement'
                )::VARCHAR
                    AS Budget_Definition,

                Optimization_Version
                    AS MILP_Input_Version

            FROM {sources["strict_summary"]}

            UNION ALL

            SELECT
                Scenario
                    AS Requirement_Scenario,

                1.0::DOUBLE
                    AS Budget_Factor,

                100.0::DOUBLE
                    AS Budget_Percent,

                B100
                    AS MILP_B100,

                B100
                    AS Scenario_Budget,

                FALSE::BOOLEAN
                    AS Budget_Is_Actual_CDH_Budget,

                (
                    'Modeled percentage of '
                    'whole-unit eligible '
                    'full-funding requirement'
                )::VARCHAR
                    AS Budget_Definition,

                Validation_VERSION
                    AS MILP_Input_Version

            FROM {sources["full_validation"]}
            """
        )

        connection.execute(
            "COMMIT"
        )

    except Exception:
        try:
            connection.execute(
                "ROLLBACK"
            )
        except Exception:
            pass

        raise

    finally:
        connection.close()


def main() -> None:

    print("=" * 100)
    print("CDH INVENTORY DSS")
    print("CURRENT PRESENTATION CONTRACT BUILDER")
    print("=" * 100)
    print()

    print(
        "Run mode:",
        RUNTIME_CONTEXT.run_mode,
    )

    print(
        "Planning year:",
        TARGET_YEAR,
    )

    print(
        "Database:",
        DATABASE_PATH,
    )

    build_contract()

    print()
    print(
        "CURRENT presentation contract: BUILT"
    )

    print(
        "Publication status: "
        "CANDIDATE_NOT_PUBLISHED"
    )

    print(
        "No live database was published."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
