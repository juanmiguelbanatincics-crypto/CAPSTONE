from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "cdh_inventory.duckdb"
)

DEFAULT_RAW_ANNUAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "annual_inventory"
)

DEFAULT_VALIDATED_ANNUAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "validated"
    / "annual_inventory"
)

DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DEFAULT_MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

VALID_RUN_MODES = {
    "BASELINE",
    "STAGING",
    "TEST",
    "CURRENT",
}


def _resolve_project_path(
    value: str | Path | None,
    default: Path,
) -> Path:
    if value is None:
        return default.resolve()

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


@dataclass(frozen=True)
class RuntimeContext:
    project_root: Path
    database_path: Path
    raw_annual_dir: Path
    validated_annual_dir: Path
    processed_dir: Path
    models_dir: Path
    historical_start_year: int
    latest_completed_year: int
    planning_year: int
    run_mode: str
    run_id: str

    @property
    def historical_years(self) -> list[int]:
        return list(
            range(
                self.historical_start_year,
                self.latest_completed_year + 1,
            )
        )

    @property
    def supervised_target_years(self) -> list[int]:
        return list(
            range(
                self.historical_start_year + 1,
                self.latest_completed_year + 1,
            )
        )

    @property
    def feature_cutoff_year(self) -> int:
        return self.latest_completed_year

    @property
    def is_baseline(self) -> bool:
        return self.run_mode == "BASELINE"

    @property
    def is_isolated_run(self) -> bool:
        return self.run_mode in {
            "STAGING",
            "TEST",
        }

    def validate(self) -> None:
        if self.latest_completed_year < self.historical_start_year:
            raise ValueError(
                "Latest completed year cannot be earlier "
                "than historical start year."
            )

        if self.planning_year != self.latest_completed_year + 1:
            raise ValueError(
                "Planning year must be exactly one year after "
                "the latest completed annual year."
            )

        if self.run_mode not in VALID_RUN_MODES:
            raise ValueError(
                f"Unsupported run mode: {self.run_mode}. "
                f"Expected one of {sorted(VALID_RUN_MODES)}."
            )

        if not self.run_id.strip():
            raise ValueError(
                "Run ID cannot be blank."
            )

        if self.is_isolated_run:
            if (
                self.database_path.resolve()
                == DEFAULT_DATABASE_PATH.resolve()
            ):
                raise ValueError(
                    "STAGING/TEST runs cannot use the live "
                    "baseline DuckDB database."
                )

            if (
                self.validated_annual_dir.resolve()
                == DEFAULT_VALIDATED_ANNUAL_DIR.resolve()
            ):
                raise ValueError(
                    "STAGING/TEST runs cannot write validated "
                    "outputs into the baseline validated folder."
                )

            if (
                self.processed_dir.resolve()
                == DEFAULT_PROCESSED_DIR.resolve()
            ):
                raise ValueError(
                    "STAGING/TEST runs cannot write processed "
                    "outputs into the baseline processed folder."
                )

            if (
                self.models_dir.resolve()
                == DEFAULT_MODELS_DIR.resolve()
            ):
                raise ValueError(
                    "STAGING/TEST runs cannot write model "
                    "artifacts into the baseline models folder."
                )

    def to_dict(self) -> dict:
        payload = asdict(self)

        for field in [
            "project_root",
            "database_path",
            "raw_annual_dir",
            "validated_annual_dir",
            "processed_dir",
            "models_dir",
        ]:
            payload[field] = str(
                payload[field]
            )

        payload[
            "historical_years"
        ] = self.historical_years

        payload[
            "supervised_target_years"
        ] = self.supervised_target_years

        payload[
            "feature_cutoff_year"
        ] = self.feature_cutoff_year

        return payload


def build_runtime_context(
    *,
    database_path: str | Path | None = None,
    raw_annual_dir: str | Path | None = None,
    validated_annual_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    historical_start_year: int = 2021,
    latest_completed_year: int = 2025,
    planning_year: int | None = None,
    run_mode: str = "BASELINE",
    run_id: str | None = None,
) -> RuntimeContext:
    resolved_planning_year = (
        latest_completed_year + 1
        if planning_year is None
        else planning_year
    )

    normalized_mode = run_mode.strip().upper()

    resolved_run_id = (
        f"{resolved_planning_year}.1-"
        f"{normalized_mode.lower()}"
        if run_id is None
        else run_id.strip()
    )

    context = RuntimeContext(
        project_root=PROJECT_ROOT,
        database_path=_resolve_project_path(
            database_path,
            DEFAULT_DATABASE_PATH,
        ),
        raw_annual_dir=_resolve_project_path(
            raw_annual_dir,
            DEFAULT_RAW_ANNUAL_DIR,
        ),
        validated_annual_dir=_resolve_project_path(
            validated_annual_dir,
            DEFAULT_VALIDATED_ANNUAL_DIR,
        ),
        processed_dir=_resolve_project_path(
            processed_dir,
            DEFAULT_PROCESSED_DIR,
        ),
        models_dir=_resolve_project_path(
            models_dir,
            DEFAULT_MODELS_DIR,
        ),
        historical_start_year=int(
            historical_start_year
        ),
        latest_completed_year=int(
            latest_completed_year
        ),
        planning_year=int(
            resolved_planning_year
        ),
        run_mode=normalized_mode,
        run_id=resolved_run_id,
    )

    context.validate()

    return context


def load_runtime_context_from_env() -> RuntimeContext:
    historical_start_year = int(
        os.getenv(
            "CDH_HISTORICAL_START_YEAR",
            "2021",
        )
    )

    latest_completed_year = int(
        os.getenv(
            "CDH_LATEST_COMPLETED_YEAR",
            "2025",
        )
    )

    planning_year_raw = os.getenv(
        "CDH_PLANNING_YEAR"
    )

    planning_year = (
        None
        if planning_year_raw is None
        else int(planning_year_raw)
    )

    return build_runtime_context(
        database_path=os.getenv(
            "CDH_DATABASE_PATH"
        ),
        raw_annual_dir=os.getenv(
            "CDH_RAW_ANNUAL_DIR"
        ),
        validated_annual_dir=os.getenv(
            "CDH_VALIDATED_ANNUAL_DIR"
        ),
        processed_dir=os.getenv(
            "CDH_PROCESSED_DIR"
        ),
        models_dir=os.getenv(
            "CDH_MODELS_DIR"
        ),
        historical_start_year=historical_start_year,
        latest_completed_year=latest_completed_year,
        planning_year=planning_year,
        run_mode=os.getenv(
            "CDH_RUN_MODE",
            "BASELINE",
        ),
        run_id=os.getenv(
            "CDH_RUN_ID"
        ),
    )
