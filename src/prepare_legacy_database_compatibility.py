"""Prepare an older 2026 DuckDB snapshot for the current read-only app.

The current Streamlit app reads a governed ``current_*`` presentation
contract. The shared handoff database predates that contract, but contains
the corresponding validated 2026 source tables. This utility creates a
project-local copy and adds compatibility views before invoking the existing
contract builder. The source database is always opened read-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "C:/Users/cedru/Downloads/cdh_inventory.duckdb"
)
DEFAULT_TARGET = (
    PROJECT_ROOT
    / "database"
    / "versions"
    / "legacy-2026-compat"
    / "cdh_inventory.duckdb"
)


def create_legacy_source_views(
    database_path: Path,
) -> None:
    """Expose older validated tables under the governed builder's names."""

    connection = duckdb.connect(str(database_path))

    try:
        connection.execute("BEGIN TRANSACTION")

        connection.execute(
            """
            CREATE OR REPLACE VIEW
            forecast_2026_governed_candidate AS
            SELECT
                *,
                Point_Forecast_2026 AS Point_Forecast,
                Hurdle_Point_Forecast_Challenger
                    AS Hurdle_Point_Forecast_Reference
            FROM forecast_2026_candidate
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW
            forecast_2026_governed_candidate_segment_summary AS
            SELECT
                Segment_Type,
                Segment,
                Medicine_Count AS Rows,
                Point_Forecast_Total,
                Hurdle_Challenger_Total AS Hurdle_Reference_Total,
                P50_Total,
                P80_Total,
                P90_Total,
                Mean_Occurrence_Probability
            FROM forecast_2026_segment_summary
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW
            forecast_2026_governed_candidate_governance AS
            SELECT *
            FROM forecast_2026_governance
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW
            planning_2026_governed_inputs AS
            SELECT
                *,
                Point_Forecast_2026 AS Point_Forecast,
                Hurdle_Point_Forecast_Challenger
                    AS Hurdle_Point_Forecast_Reference
            FROM planning_2026_inputs
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW
            planning_2026_governed_governance AS
            SELECT *
            FROM planning_2026_governance
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW
            planning_2026_governed_review_queue AS
            SELECT
                *,
                High_Priority_ABC_Review AS High_Priority_AB_Unresolved
            FROM planning_2026_review_queue
            """
        )

        direct_views = {
            "planning_2026_governed_milp_governance":
                "planning_2026_milp_governance",
            "planning_2026_governed_milp_strict_summary":
                "planning_2026_milp_strict_summary",
            "planning_2026_governed_milp_strict_by_abc":
                "planning_2026_milp_strict_by_abc",
            "planning_2026_governed_milp_strict_allocations":
                "planning_2026_milp_strict_allocations",
            "planning_2026_governed_milp_strict_priority_audit":
                "planning_2026_milp_strict_priority_audit",
            "planning_2026_governed_milp_strict_governance":
                "planning_2026_milp_strict_governance",
            "planning_2026_governed_milp_full_funding_validation":
                "planning_2026_milp_full_funding_validation",
            "planning_2026_governed_milp_full_funding_by_abc":
                "planning_2026_milp_full_funding_by_abc",
            "planning_2026_governed_milp_full_funding_allocations":
                "planning_2026_milp_full_funding_allocations",
        }

        for target, source in direct_views.items():
            connection.execute(
                f"""
                CREATE OR REPLACE VIEW {target} AS
                SELECT *
                FROM {source}
                """
            )

        connection.execute(
            """
            CREATE OR REPLACE VIEW
            planning_2026_governed_milp_full_funding_governance AS
            SELECT
                Validation_VERSION AS Compatibility_Version,
                'LEGACY_2026_SNAPSHOT' AS Source_Snapshot,
                'Derived locally for the current presentation contract'
                    AS Compatibility_Status,
                COUNT(*)::BIGINT AS Validated_Scenario_Count,
                SUM(
                    CASE
                        WHEN Exact_Full_Allocation_PASS
                        THEN 1 ELSE 0
                    END
                )::BIGINT AS Exact_Full_Allocation_Pass_Count
            FROM planning_2026_milp_full_funding_validation
            GROUP BY Validation_VERSION
            """
        )

        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def build_current_contract(
    database_path: Path,
) -> None:
    """Run the repository's governed presentation-contract builder."""

    runtime_root = (
        PROJECT_ROOT
        / "data"
        / "local_runtime"
    )

    os.environ.update(
        {
            "CDH_DATABASE_PATH": str(database_path),
            "CDH_RUN_MODE": "STAGING",
            "CDH_RUN_ID": "legacy-2026-local-compat",
            "CDH_VALIDATED_ANNUAL_DIR": str(
                runtime_root / "validated"
            ),
            "CDH_PROCESSED_DIR": str(
                runtime_root / "processed"
            ),
            "CDH_MODELS_DIR": str(runtime_root / "models"),
        }
    )

    from build_current_presentation_contract import build_contract

    build_contract()


def write_local_current_pointer(
    database_path: Path,
) -> None:
    """Activate the versioned local compatibility snapshot for the app."""

    database_dir = PROJECT_ROOT / "database"
    relative_path = database_path.relative_to(database_dir)
    pointer_path = database_dir / "current.json"

    pointer_path.write_text(
        json.dumps(
            {
                "Pointer_Version": "CURRENT_DATABASE_POINTER_V1",
                "Database_Relative_Path": str(relative_path).replace(
                    "\\",
                    "/",
                ),
                "Local_Compatibility_Mode": True,
                "Source_Database": "cdh_inventory.duckdb",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a project-local DuckDB compatibility "
            "database for the current Streamlit app."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Read-only legacy DuckDB source path.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Project-local compatible DuckDB target path.",
    )
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    target = arguments.target.resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Legacy source database not found: {source}"
        )

    if source == target:
        raise ValueError(
            "The compatibility target must not overwrite "
            "the source database."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    create_legacy_source_views(target)
    build_current_contract(target)
    write_local_current_pointer(target)

    print("Compatibility database created:")
    print(target)


if __name__ == "__main__":
    main()
