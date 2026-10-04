from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = PROJECT_ROOT / "database"

VERSIONS_DIR = (
    DATABASE_DIR / "versions"
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


def validate_version(
    version: str,
) -> None:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}",
        version,
    ):
        raise ValueError(
            "Invalid version identifier."
        )


def snapshot_paths(
    version: str,
) -> tuple[Path, Path]:
    validate_version(
        version
    )

    version_dir = (
        VERSIONS_DIR / version
    ).resolve()

    try:
        version_dir.relative_to(
            VERSIONS_DIR.resolve()
        )
    except ValueError as error:
        raise RuntimeError(
            "Version resolves outside "
            "database/versions."
        ) from error

    snapshot = (
        version_dir
        / "cdh_inventory.duckdb"
    )

    manifest = (
        version_dir
        / "manifest.json"
    )

    return snapshot, manifest


def validate_published_snapshot(
    version: str,
) -> dict:
    snapshot, manifest_path = (
        snapshot_paths(
            version
        )
    )

    if not snapshot.exists():
        raise FileNotFoundError(
            f"Snapshot not found: {snapshot}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        manifest.get(
            "Manifest_Version"
        )
        != "GOVERNED_PUBLICATION_MANIFEST_V1"
    ):
        raise RuntimeError(
            "Unsupported publication manifest."
        )

    if (
        manifest.get(
            "Published_Version"
        )
        != version
    ):
        raise RuntimeError(
            "Manifest version does not match "
            "requested version."
        )

    expected_hash = manifest.get(
        "Published_Snapshot_SHA256"
    )

    actual_hash = sha256_file(
        snapshot
    )

    if (
        not expected_hash
        or actual_hash
        != expected_hash
    ):
        raise RuntimeError(
            "Published snapshot SHA-256 "
            "verification failed."
        )

    connection = duckdb.connect(
        str(snapshot),
        read_only=True,
    )

    try:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }

        missing = (
            REQUIRED_CURRENT_OBJECTS
            - tables
        )

        if missing:
            raise RuntimeError(
                "Published snapshot is missing "
                "required current objects: "
                + ", ".join(
                    sorted(missing)
                )
            )

        context = connection.execute(
            """
            SELECT
                Planning_Year,
                Candidate_Run_ID,
                Current_State,
                Published_Version,
                Publication_Status,
                Automatic_Publication
            FROM current_system_context
            """
        ).fetchone()

        if context is None:
            raise RuntimeError(
                "Published current_system_context "
                "is empty."
            )

        (
            planning_year,
            candidate_run_id,
            current_state,
            published_version,
            publication_status,
            automatic_publication,
        ) = context

        if (
            current_state
            != "PUBLISHED_CURRENT"
            or published_version
            != version
            or publication_status
            != "PUBLISHED"
            or automatic_publication
            is not False
        ):
            raise RuntimeError(
                "Snapshot does not have valid "
                "published CURRENT metadata."
            )

        if (
            manifest.get(
                "Planning_Year"
            )
            != planning_year
        ):
            raise RuntimeError(
                "Manifest planning year does not "
                "match the snapshot."
            )

        if (
            manifest.get(
                "Candidate_Run_ID"
            )
            != candidate_run_id
        ):
            raise RuntimeError(
                "Manifest candidate run does not "
                "match the snapshot."
            )

        registry = connection.execute(
            """
            SELECT
                Published_Version,
                Candidate_Run_ID,
                Planning_Year,
                Published_Snapshot_SHA256
            FROM (
                SELECT
                    Published_Version,
                    Candidate_Run_ID,
                    Planning_Year,
                    NULL::VARCHAR
                        AS Published_Snapshot_SHA256
                FROM publication_registry
            )
            """
        ).fetchone()

        if registry is None:
            raise RuntimeError(
                "publication_registry is empty."
            )

        if (
            registry[0] != version
            or registry[1]
                != candidate_run_id
            or registry[2]
                != planning_year
        ):
            raise RuntimeError(
                "Publication registry does not "
                "match the snapshot."
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

        if (
            strict[0] <= 0
            or strict[0]
                != strict[1]
        ):
            raise RuntimeError(
                "Strict ABC priority audit "
                "does not fully pass."
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

        if (
            full[0] <= 0
            or full[1] != 0
            or full[0]
                != full[2]
        ):
            raise RuntimeError(
                "Full-funding validation "
                "does not fully pass."
            )

        return {
            "Version":
                version,
            "Planning_Year":
                int(planning_year),
            "Candidate_Run_ID":
                candidate_run_id,
            "Snapshot_SHA256":
                actual_hash,
            "Snapshot":
                snapshot,
        }

    finally:
        connection.close()


def read_existing_pointer() -> dict | None:
    if not CURRENT_POINTER_PATH.exists():
        return None

    payload = json.loads(
        CURRENT_POINTER_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get(
            "Pointer_Version"
        )
        != POINTER_VERSION
    ):
        raise RuntimeError(
            "Existing CURRENT pointer has "
            "an unsupported version."
        )

    return payload


def atomic_write_pointer(
    payload: dict,
) -> None:
    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        DATABASE_DIR
        / (
            f".current-{uuid.uuid4().hex}.tmp"
        )
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )

            handle.write(
                "\n"
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary,
            CURRENT_POINTER_PATH,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


def switch_current(
    version: str,
    action: str,
) -> None:
    validated = (
        validate_published_snapshot(
            version
        )
    )

    previous = (
        read_existing_pointer()
    )

    relative_path = str(
        validated[
            "Snapshot"
        ].relative_to(
            DATABASE_DIR.resolve()
        )
    ).replace(
        "\\",
        "/",
    )

    payload = {
        "Pointer_Version":
            POINTER_VERSION,

        "Database_Relative_Path":
            relative_path,

        "Active_Version":
            version,

        "Planning_Year":
            validated[
                "Planning_Year"
            ],

        "Candidate_Run_ID":
            validated[
                "Candidate_Run_ID"
            ],

        "Snapshot_SHA256":
            validated[
                "Snapshot_SHA256"
            ],

        "Activated_At_UTC":
            datetime.now(
                timezone.utc
            )
            .replace(
                microsecond=0
            )
            .isoformat(),

        "Activation_Action":
            action,

        "Activation_Git_Commit":
            git_head(),

        "Previous_Active_Version":
            (
                previous.get(
                    "Active_Version"
                )
                if previous
                else None
            ),

        "Automatic_Publication":
            False,
    }

    atomic_write_pointer(
        payload
    )

    written = read_existing_pointer()

    if written != payload:
        raise RuntimeError(
            "Atomic CURRENT pointer "
            "post-write verification failed."
        )

    resolved = (
        DATABASE_DIR
        / written[
            "Database_Relative_Path"
        ]
    ).resolve()

    if (
        resolved
        != validated[
            "Snapshot"
        ].resolve()
    ):
        raise RuntimeError(
            "CURRENT pointer resolved to "
            "an unexpected snapshot."
        )

    if (
        sha256_file(
            resolved
        )
        != validated[
            "Snapshot_SHA256"
        ]
    ):
        raise RuntimeError(
            "CURRENT snapshot hash changed "
            "after pointer activation."
        )

    print(
        "=" * 88
    )
    print(
        "CDH INVENTORY DSS"
    )
    print(
        "ATOMIC CURRENT DATABASE SWITCH"
    )
    print(
        "=" * 88
    )
    print(
        f"Action: {action}"
    )
    print(
        f"Active version: {version}"
    )
    print(
        "Planning year: "
        f"{validated['Planning_Year']}"
    )
    print(
        "CURRENT database:"
    )
    print(
        f"  {resolved}"
    )
    print(
        "Snapshot SHA256:"
    )
    print(
        "  "
        f"{validated['Snapshot_SHA256']}"
    )
    print(
        "Pointer version:"
    )
    print(
        f"  {POINTER_VERSION}"
    )
    print(
        "Atomic os.replace switch: PASS"
    )
    print(
        "Snapshot post-switch integrity: PASS"
    )
    print(
        "Automatic publication: FALSE"
    )
    print(
        "=" * 88
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically activate or roll back "
            "the CDH CURRENT database pointer."
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    activate_parser = (
        subparsers.add_parser(
            "activate"
        )
    )

    activate_parser.add_argument(
        "--version",
        required=True,
    )

    rollback_parser = (
        subparsers.add_parser(
            "rollback"
        )
    )

    rollback_parser.add_argument(
        "--version",
        required=True,
    )

    args = parser.parse_args()

    if args.command == "activate":
        switch_current(
            version=args.version,
            action="ACTIVATE",
        )

    elif args.command == "rollback":
        switch_current(
            version=args.version,
            action="ROLLBACK",
        )


if __name__ == "__main__":
    main()
