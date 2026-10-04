from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = PROJECT_ROOT / "database"
STAGING_DIR = DATABASE_DIR / "staging"
VERSIONS_DIR = DATABASE_DIR / "versions"
BACKUPS_DIR = DATABASE_DIR / "backups"

DEFAULT_DATABASE_PATH = (
    DATABASE_DIR / "cdh_inventory.duckdb"
)

CURRENT_POINTER_PATH = (
    DATABASE_DIR / "current.json"
)

POINTER_VERSION = (
    "CURRENT_DATABASE_POINTER_V1"
)

REQUIRED_CURRENT_OBJECTS = {
    "current_system_context",
    "current_forecast",
    "current_forecast_segment_summary",
    "current_forecast_governance",
    "current_planning_inputs",
    "current_planning_governance",
    "current_review_queue",
    "current_review_summary",
    "current_milp_budget_scenarios",
    "current_milp_governance",
    "current_milp_strict_summary",
    "current_milp_strict_by_abc",
    "current_milp_strict_allocations",
    "current_milp_strict_priority_audit",
    "current_milp_strict_governance",
    "current_milp_full_funding_validation",
    "current_milp_full_funding_by_abc",
    "current_milp_full_funding_allocations",
    "current_milp_full_funding_governance",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_head() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except Exception:
        return "UNKNOWN"


def safe_relative(
    path: Path,
    root: Path,
) -> str:
    return str(
        path.resolve().relative_to(
            root.resolve()
        )
    ).replace("\\", "/")


def get_tables(
    connection: duckdb.DuckDBPyConnection,
) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        ).fetchall()
    }


def get_columns(
    connection: duckdb.DuckDBPyConnection,
    table: str,
) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            f'DESCRIBE "{table}"'
        ).fetchall()
    }


def validate_version(version: str) -> None:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
        version,
    ):
        raise ValueError(
            "Invalid version identifier."
        )


def resolve_active_database() -> tuple[
    Path,
    dict | None,
]:
    if not CURRENT_POINTER_PATH.exists():
        if not DEFAULT_DATABASE_PATH.exists():
            raise FileNotFoundError(
                "No CURRENT pointer exists and "
                "legacy live database was not found."
            )

        return (
            DEFAULT_DATABASE_PATH.resolve(),
            None,
        )

    payload = json.loads(
        CURRENT_POINTER_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get("Pointer_Version")
        != POINTER_VERSION
    ):
        raise RuntimeError(
            "Unsupported CURRENT pointer version."
        )

    relative_value = payload.get(
        "Database_Relative_Path"
    )

    if (
        not isinstance(
            relative_value,
            str,
        )
        or not relative_value.strip()
    ):
        raise RuntimeError(
            "CURRENT pointer is missing "
            "Database_Relative_Path."
        )

    relative_path = Path(
        relative_value
    )

    if relative_path.is_absolute():
        raise RuntimeError(
            "CURRENT pointer must use "
            "a relative path."
        )

    active = (
        DATABASE_DIR
        / relative_path
    ).resolve()

    try:
        active.relative_to(
            VERSIONS_DIR.resolve()
        )
    except ValueError as error:
        raise RuntimeError(
            "CURRENT pointer resolves outside "
            "database/versions."
        ) from error

    if (
        active.suffix.lower()
        != ".duckdb"
    ):
        raise RuntimeError(
            "CURRENT pointer does not reference "
            "a DuckDB snapshot."
        )

    if not active.exists():
        raise FileNotFoundError(
            f"CURRENT database not found: {active}"
        )

    return active, payload


def validate_candidate(
    source: Path,
) -> dict:
    source = source.resolve()

    try:
        source.relative_to(
            STAGING_DIR.resolve()
        )
    except ValueError as error:
        raise RuntimeError(
            "Publication source must be inside "
            "database/staging."
        ) from error

    if not source.exists():
        raise FileNotFoundError(
            f"Candidate database not found: {source}"
        )

    connection = duckdb.connect(
        str(source),
        read_only=True,
    )

    try:
        tables = get_tables(
            connection
        )

        missing = (
            REQUIRED_CURRENT_OBJECTS
            - tables
        )

        if missing:
            raise RuntimeError(
                "Candidate is missing current "
                "presentation objects: "
                + ", ".join(
                    sorted(missing)
                )
            )

        context = connection.execute(
            """
            SELECT
                Contract_Version,
                Historical_Start_Year,
                Latest_Completed_Year,
                Planning_Year,
                Candidate_Run_ID,
                Build_Run_Mode,
                Current_State,
                Published_Version,
                Publication_Status,
                Automatic_Publication
            FROM current_system_context
            """
        ).fetchone()

        if context is None:
            raise RuntimeError(
                "current_system_context is empty."
            )

        (
            contract_version,
            historical_start_year,
            latest_completed_year,
            planning_year,
            candidate_run_id,
            build_run_mode,
            current_state,
            published_version,
            publication_status,
            automatic_publication,
        ) = context

        if build_run_mode != "STAGING":
            raise RuntimeError(
                "Only a STAGING candidate "
                "may be published."
            )

        if (
            current_state
            != "VALIDATED_CANDIDATE"
        ):
            raise RuntimeError(
                "Candidate Current_State is not "
                "VALIDATED_CANDIDATE."
            )

        if published_version is not None:
            raise RuntimeError(
                "Candidate already contains a "
                "Published_Version."
            )

        if (
            publication_status
            != "CANDIDATE_NOT_PUBLISHED"
        ):
            raise RuntimeError(
                "Candidate publication status is "
                "not CANDIDATE_NOT_PUBLISHED."
            )

        if automatic_publication is not False:
            raise RuntimeError(
                "Automatic publication must be FALSE."
            )

        model_gov = connection.execute(
            """
            SELECT
                Run_Mode,
                Run_ID,
                Planning_Year,
                Recommended_Model,
                Automatic_Promotion_Allowed,
                Publication_Status
            FROM forecast_model_selection_governance
            """
        ).fetchone()

        if (
            model_gov[0] != "STAGING"
            or model_gov[1]
                != candidate_run_id
            or model_gov[2]
                != planning_year
            or model_gov[4] is not False
            or model_gov[5]
                != "STAGING_RECOMMENDATION_PENDING_PUBLICATION"
        ):
            raise RuntimeError(
                "Forecast model governance is not "
                "a publishable STAGING recommendation."
            )

        strict = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(
                    CASE
                        WHEN Strict_ABC_Priority_PASS
                        THEN 1
                        ELSE 0
                    END
                )
            FROM current_milp_strict_priority_audit
            """
        ).fetchone()

        if strict != (12, 12):
            raise RuntimeError(
                "Strict ABC priority audit "
                "did not pass 12/12."
            )

        full = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(
                    Full_Allocation_Mismatch_Count
                ),
                SUM(
                    CASE
                        WHEN Exact_Full_Allocation_PASS
                        THEN 1
                        ELSE 0
                    END
                )
            FROM
                current_milp_full_funding_validation
            """
        ).fetchone()

        if full != (3, 0, 3):
            raise RuntimeError(
                "Full-funding validation "
                "did not pass 3/3."
            )

        strict_gov = connection.execute(
            """
            SELECT
                Run_Mode,
                Publication_Status,
                Automatic_Publication
            FROM current_milp_strict_governance
            """
        ).fetchone()

        if (
            strict_gov[0] != "STAGING"
            or strict_gov[1]
                != "CANDIDATE_NOT_PUBLISHED"
            or strict_gov[2] is not False
        ):
            raise RuntimeError(
                "Strict MILP governance is not "
                "a valid unpublished STAGING result."
            )

        full_gov = connection.execute(
            """
            SELECT
                Run_Mode,
                Publication_Status,
                Automatic_Publication
            FROM current_milp_full_funding_governance
            """
        ).fetchone()

        if (
            full_gov[0] != "STAGING"
            or full_gov[1]
                != "CANDIDATE_NOT_PUBLISHED"
            or full_gov[2] is not False
        ):
            raise RuntimeError(
                "Full-funding governance is not "
                "a valid unpublished STAGING result."
            )

        return {
            "Contract_Version":
                contract_version,
            "Historical_Start_Year":
                int(
                    historical_start_year
                ),
            "Latest_Completed_Year":
                int(
                    latest_completed_year
                ),
            "Planning_Year":
                int(
                    planning_year
                ),
            "Candidate_Run_ID":
                candidate_run_id,
            "Recommended_Model":
                model_gov[3],
        }

    finally:
        connection.close()


def set_published_metadata(
    snapshot: Path,
    version: str,
    candidate: dict,
    source_relative: str,
    source_sha256: str,
    previous_relative: str,
    previous_sha256: str,
    published_at_utc: str,
    commit: str,
) -> None:
    connection = duckdb.connect(
        str(snapshot)
    )

    try:
        connection.execute(
            "BEGIN TRANSACTION"
        )

        planning_year = (
            candidate["Planning_Year"]
        )

        run_id = (
            candidate["Candidate_Run_ID"]
        )

        connection.execute(
            """
            UPDATE current_system_context
            SET
                Current_State =
                    'PUBLISHED_CURRENT',
                Published_Version = ?,
                Publication_Status =
                    'PUBLISHED',
                Automatic_Publication =
                    FALSE
            """,
            [version],
        )

        connection.execute(
            """
            UPDATE
                forecast_model_selection_governance
            SET
                Publication_Status =
                    'PUBLISHED'
            WHERE
                Run_ID = ?
                AND Planning_Year = ?
            """,
            [
                run_id,
                planning_year,
            ],
        )

        connection.execute(
            """
            UPDATE
                forecast_model_selection_governance_history
            SET
                Publication_Status =
                    'PUBLISHED'
            WHERE
                Run_ID = ?
                AND Planning_Year = ?
            """,
            [
                run_id,
                planning_year,
            ],
        )

        tables = get_tables(
            connection
        )

        # Publish only writable governed base tables.
        # The current_milp_* governance objects are
        # presentation views over these tables and
        # therefore inherit the published metadata.
        publication_tables = [
            (
                f"planning_{planning_year}_"
                "governed_milp_strict_governance"
            ),
            (
                f"planning_{planning_year}_"
                "governed_milp_full_funding_governance"
            ),
        ]

        for table in publication_tables:
            if table not in tables:
                continue

            columns = get_columns(
                connection,
                table,
            )

            if (
                "Publication_Status"
                in columns
            ):
                connection.execute(
                    f"""
                    UPDATE "{table}"
                    SET Publication_Status =
                        'PUBLISHED'
                    """
                )

            if (
                "Automatic_Publication"
                in columns
            ):
                connection.execute(
                    f"""
                    UPDATE "{table}"
                    SET Automatic_Publication =
                        FALSE
                    """
                )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                publication_registry
            (
                Published_Version VARCHAR,
                Candidate_Run_ID VARCHAR,
                Planning_Year BIGINT,
                Published_At_UTC VARCHAR,
                Source_Staging_Relative_Path VARCHAR,
                Source_SHA256 VARCHAR,
                Previous_Active_Relative_Path VARCHAR,
                Previous_Active_SHA256 VARCHAR,
                Code_Git_Commit VARCHAR,
                Pointer_Version VARCHAR,
                Publication_Method VARCHAR,
                Automatic_Publication BOOLEAN
            )
            """
        )

        connection.execute(
            """
            INSERT INTO publication_registry
            VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, FALSE
            )
            """,
            [
                version,
                run_id,
                planning_year,
                published_at_utc,
                source_relative,
                source_sha256,
                previous_relative,
                previous_sha256,
                commit,
                POINTER_VERSION,
                "HUMAN_APPROVED_ATOMIC_POINTER_PUBLICATION",
            ],
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


def prepare(
    source: Path,
    version: str,
) -> None:
    validate_version(
        version
    )

    source = source.resolve()

    candidate = validate_candidate(
        source
    )

    final_version_dir = (
        VERSIONS_DIR / version
    )

    final_snapshot = (
        final_version_dir
        / "cdh_inventory.duckdb"
    )

    final_backup_dir = (
        BACKUPS_DIR
        / f"before-{version}"
    )

    if final_version_dir.exists():
        raise FileExistsError(
            "Version already exists: "
            f"{final_version_dir}"
        )

    if final_backup_dir.exists():
        raise FileExistsError(
            "Backup already exists: "
            f"{final_backup_dir}"
        )

    active_db, pointer_payload = (
        resolve_active_database()
    )

    source_sha256 = sha256_file(
        source
    )

    previous_sha256 = sha256_file(
        active_db
    )

    source_relative = safe_relative(
        source,
        PROJECT_ROOT,
    )

    previous_relative = safe_relative(
        active_db,
        PROJECT_ROOT,
    )

    pointer_before = (
        CURRENT_POINTER_PATH.read_bytes()
        if CURRENT_POINTER_PATH.exists()
        else None
    )

    commit = git_head()

    published_at_utc = (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )

    temp_version_dir = (
        VERSIONS_DIR
        / (
            f".{version}.prepare-"
            f"{uuid.uuid4().hex}"
        )
    )

    temp_backup_dir = (
        BACKUPS_DIR
        / (
            f".before-{version}.prepare-"
            f"{uuid.uuid4().hex}"
        )
    )

    VERSIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BACKUPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        temp_version_dir.mkdir(
            parents=False
        )

        temp_backup_dir.mkdir(
            parents=False
        )

        temp_snapshot = (
            temp_version_dir
            / "cdh_inventory.duckdb"
        )

        temp_backup = (
            temp_backup_dir
            / "cdh_inventory.duckdb"
        )

        shutil.copy2(
            source,
            temp_snapshot,
        )

        shutil.copy2(
            active_db,
            temp_backup,
        )

        if (
            sha256_file(
                temp_snapshot
            )
            != source_sha256
        ):
            raise RuntimeError(
                "Initial snapshot copy hash mismatch."
            )

        if (
            sha256_file(
                temp_backup
            )
            != previous_sha256
        ):
            raise RuntimeError(
                "Backup copy hash mismatch."
            )

        set_published_metadata(
            snapshot=temp_snapshot,
            version=version,
            candidate=candidate,
            source_relative=source_relative,
            source_sha256=source_sha256,
            previous_relative=previous_relative,
            previous_sha256=previous_sha256,
            published_at_utc=published_at_utc,
            commit=commit,
        )

        snapshot_sha256 = (
            sha256_file(
                temp_snapshot
            )
        )

        backup_sha256 = (
            sha256_file(
                temp_backup
            )
        )

        manifest = {
            "Manifest_Version":
                "GOVERNED_PUBLICATION_MANIFEST_V1",
            "Published_Version":
                version,
            "Candidate_Run_ID":
                candidate[
                    "Candidate_Run_ID"
                ],
            "Planning_Year":
                candidate[
                    "Planning_Year"
                ],
            "Published_At_UTC":
                published_at_utc,
            "Source_Staging_Relative_Path":
                source_relative,
            "Source_SHA256":
                source_sha256,
            "Published_Snapshot_SHA256":
                snapshot_sha256,
            "Previous_Active_Relative_Path":
                previous_relative,
            "Previous_Active_SHA256":
                previous_sha256,
            "Backup_Relative_Path":
                safe_relative(
                    temp_backup,
                    PROJECT_ROOT,
                ).replace(
                    temp_backup_dir.name,
                    final_backup_dir.name,
                ),
            "Backup_SHA256":
                backup_sha256,
            "Code_Git_Commit":
                commit,
            "Pointer_Version":
                POINTER_VERSION,
            "Pointer_Before":
                pointer_payload,
            "Automatic_Publication":
                False,
        }

        (
            temp_version_dir
            / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            temp_backup_dir
            / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temp_version_dir.rename(
            final_version_dir
        )

        temp_backup_dir.rename(
            final_backup_dir
        )

        if (
            sha256_file(source)
            != source_sha256
        ):
            raise RuntimeError(
                "STAGING source changed during "
                "snapshot preparation."
            )

        pointer_after = (
            CURRENT_POINTER_PATH.read_bytes()
            if CURRENT_POINTER_PATH.exists()
            else None
        )

        if pointer_after != pointer_before:
            raise RuntimeError(
                "CURRENT pointer changed during "
                "snapshot preparation."
            )

        print(
            "=" * 88
        )
        print(
            "CDH INVENTORY DSS"
        )
        print(
            "GOVERNED PUBLICATION SNAPSHOT PREPARED"
        )
        print(
            "=" * 88
        )
        print(
            f"Version: {version}"
        )
        print(
            "Planning year: "
            f"{candidate['Planning_Year']}"
        )
        print(
            "Candidate run: "
            f"{candidate['Candidate_Run_ID']}"
        )
        print(
            "Recommended model: "
            f"{candidate['Recommended_Model']}"
        )
        print(
            "Snapshot:"
        )
        print(
            f"  {final_snapshot}"
        )
        print(
            "Snapshot SHA256:"
        )
        print(
            f"  {snapshot_sha256}"
        )
        print(
            "Pre-publication backup:"
        )
        print(
            "  "
            f"{final_backup_dir / 'cdh_inventory.duckdb'}"
        )
        print(
            "Backup SHA256:"
        )
        print(
            f"  {backup_sha256}"
        )
        print(
            "STAGING source unchanged: PASS"
        )
        print(
            "CURRENT pointer unchanged: PASS"
        )
        print(
            "Snapshot activation performed: FALSE"
        )
        print(
            "=" * 88
        )

    except Exception:
        if temp_version_dir.exists():
            shutil.rmtree(
                temp_version_dir,
                ignore_errors=True,
            )

        if temp_backup_dir.exists():
            shutil.rmtree(
                temp_backup_dir,
                ignore_errors=True,
            )

        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a governed immutable "
            "publication snapshot without "
            "changing current.json."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--version",
        required=True,
    )

    args = parser.parse_args()

    prepare(
        source=args.source,
        version=args.version,
    )


if __name__ == "__main__":
    main()
