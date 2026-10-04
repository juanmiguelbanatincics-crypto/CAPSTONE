from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "raw_profile"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 hash without modifying the source file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def infer_year_from_filename(filename: str) -> int | None:
    """Extract a likely 4-digit year from a filename."""
    match = re.search(r"\b(20\d{2})\b", filename)

    if match:
        return int(match.group(1))

    return None


def infer_source_type(path: Path) -> str:
    """Classify source according to the controlled raw-data folders."""
    parent = path.parent.name.lower()

    mapping = {
        "annual_inventory": "Completed Annual Inventory",
        "year_end_inventory": "Year-End Inventory / Supporting Source",
        "expiry": "Expiry / Lot Source",
        "other": "Other Raw Source",
    }

    return mapping.get(parent, "Unclassified Raw Source")


def detect_csv_format(path: Path) -> tuple[str, str]:
    """
    Detect probable text encoding and delimiter.

    Returns:
        encoding, delimiter
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, errors="strict") as file:
                sample = file.read(65536)

            try:
                dialect = csv.Sniffer().sniff(
                    sample,
                    delimiters=",;\t|"
                )
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

            return encoding, delimiter

        except UnicodeDecodeError:
            continue

    return "latin1", ","


def read_csv_safely(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read CSV using detected encoding and delimiter."""
    encoding, delimiter = detect_csv_format(path)

    dataframe = pd.read_csv(
        path,
        encoding=encoding,
        sep=delimiter,
        low_memory=False,
    )

    metadata = {
        "encoding": encoding,
        "delimiter": repr(delimiter),
    }

    return dataframe, metadata


def normalize_column_name(value: Any) -> str:
    """Convert raw column labels to readable strings for profiling only."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def identify_possible_medicine_column(columns: list[str]) -> str | None:
    """
    Identify likely medicine/item identifier field.

    This does NOT change the dataset.
    It only helps estimate unique medicine counts.
    """
    preferred_patterns = [
        "item_id",
        "item id",
        "item_key",
        "item key",
        "item_code",
        "item code",
        "drug_id",
        "drug id",
        "medicine_id",
        "medicine id",
        "brand_name",
        "brand name",
        "item_description",
        "item description",
    ]

    normalized_lookup = {
        col.lower().strip(): col
        for col in columns
    }

    for pattern in preferred_patterns:
        if pattern in normalized_lookup:
            return normalized_lookup[pattern]

    for col in columns:
        lower = col.lower()

        if "item" in lower and (
            "id" in lower
            or "code" in lower
            or "description" in lower
        ):
            return col

    return None


def basic_dataframe_profile(
    dataframe: pd.DataFrame,
    source_file: str,
    sheet_name: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create dataframe-level and column-level metadata."""

    clean_columns = [
        normalize_column_name(column)
        for column in dataframe.columns
    ]

    dataframe = dataframe.copy()
    dataframe.columns = clean_columns

    medicine_column = identify_possible_medicine_column(clean_columns)

    unique_medicine_count = None

    if medicine_column is not None:
        unique_medicine_count = int(
            dataframe[medicine_column]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )

    dataframe_summary = {
        "source_file": source_file,
        "sheet_name": sheet_name or "",
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "duplicate_full_rows": int(dataframe.duplicated().sum()),
        "probable_medicine_column": medicine_column or "",
        "unique_medicine_count": unique_medicine_count,
    }

    column_profiles: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]

        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())

        if len(series) > 0:
            missing_pct = round(
                (missing_count / len(series)) * 100,
                4
            )
        else:
            missing_pct = 0.0

        unique_count = int(series.nunique(dropna=True))

        numeric_series = pd.to_numeric(series, errors="coerce")

        numeric_count = int(numeric_series.notna().sum())

        numeric_min = None
        numeric_max = None

        if numeric_count > 0:
            numeric_min = float(numeric_series.min())
            numeric_max = float(numeric_series.max())

        sample_values = (
            series.dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        column_profiles.append(
            {
                "source_file": source_file,
                "sheet_name": sheet_name or "",
                "column_name": column,
                "pandas_dtype": str(series.dtype),
                "row_count": int(len(series)),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "numeric_value_count": numeric_count,
                "numeric_min": numeric_min,
                "numeric_max": numeric_max,
                "sample_values": json.dumps(
                    sample_values,
                    ensure_ascii=False,
                ),
            }
        )

    return dataframe_summary, column_profiles


# ============================================================
# FILE PROFILING
# ============================================================

def profile_csv(path: Path) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    dataframe, read_metadata = read_csv_safely(path)

    dataframe_summary, column_profiles = basic_dataframe_profile(
        dataframe,
        source_file=path.name,
    )

    file_extra = {
        **read_metadata,
        "excel_sheet_count": None,
        "excel_sheet_names": "",
    }

    return file_extra, [dataframe_summary], column_profiles


def profile_excel(path: Path) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    workbook = pd.ExcelFile(path, engine="openpyxl")

    sheet_summaries: list[dict[str, Any]] = []
    column_profiles: list[dict[str, Any]] = []

    for sheet_name in workbook.sheet_names:
        dataframe = pd.read_excel(
            path,
            sheet_name=sheet_name,
            engine="openpyxl",
        )

        dataframe_summary, sheet_columns = basic_dataframe_profile(
            dataframe,
            source_file=path.name,
            sheet_name=sheet_name,
        )

        sheet_summaries.append(dataframe_summary)
        column_profiles.extend(sheet_columns)

    file_extra = {
        "encoding": "",
        "delimiter": "",
        "excel_sheet_count": len(workbook.sheet_names),
        "excel_sheet_names": json.dumps(
            workbook.sheet_names,
            ensure_ascii=False,
        ),
    }

    return file_extra, sheet_summaries, column_profiles


def profile_file(path: Path) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    file_summary = {
        "source_file": path.name,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "source_category": path.parent.name,
        "probable_source_type": infer_source_type(path),
        "file_extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "filename_year": infer_year_from_filename(path.name),
        "read_status": "PASS",
        "read_warning_or_error": "",
    }

    sheet_summaries: list[dict[str, Any]] = []
    column_profiles: list[dict[str, Any]] = []

    try:
        if path.suffix.lower() == ".csv":
            extra, sheet_summaries, column_profiles = profile_csv(path)

        elif path.suffix.lower() in {".xlsx", ".xlsm"}:
            extra, sheet_summaries, column_profiles = profile_excel(path)

        else:
            extra = {
                "encoding": "",
                "delimiter": "",
                "excel_sheet_count": None,
                "excel_sheet_names": "",
            }

            file_summary["read_status"] = "SKIPPED"
            file_summary["read_warning_or_error"] = (
                f"Unsupported profiling format: {path.suffix}"
            )

        file_summary.update(extra)

    except Exception as error:
        file_summary.update(
            {
                "encoding": "",
                "delimiter": "",
                "excel_sheet_count": None,
                "excel_sheet_names": "",
            }
        )

        file_summary["read_status"] = "FAIL"
        file_summary["read_warning_or_error"] = (
            f"{type(error).__name__}: {error}"
        )

    return file_summary, sheet_summaries, column_profiles


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 70)
    print("CDH INVENTORY DSS — RAW DATA PROFILER")
    print("=" * 70)
    print(f"Raw data directory: {RAW_DIR}")
    print()

    raw_files = sorted(
        [
            path
            for path in RAW_DIR.rglob("*")
            if path.is_file()
        ]
    )

    if not raw_files:
        raise FileNotFoundError(
            f"No raw files found under: {RAW_DIR}"
        )

    print(f"Raw files detected: {len(raw_files)}")
    print()

    file_summaries: list[dict[str, Any]] = []
    sheet_summaries: list[dict[str, Any]] = []
    column_profiles: list[dict[str, Any]] = []

    for index, path in enumerate(raw_files, start=1):
        print(
            f"[{index}/{len(raw_files)}] Profiling "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

        file_summary, file_sheets, file_columns = profile_file(path)

        file_summaries.append(file_summary)
        sheet_summaries.extend(file_sheets)
        column_profiles.extend(file_columns)

        print(
            f"    Status: {file_summary['read_status']}"
        )

        if file_summary["read_warning_or_error"]:
            print(
                f"    Warning/Error: "
                f"{file_summary['read_warning_or_error']}"
            )

    files_df = pd.DataFrame(file_summaries)
    sheets_df = pd.DataFrame(sheet_summaries)
    columns_df = pd.DataFrame(column_profiles)

    files_output = OUTPUT_DIR / "raw_file_inventory.csv"
    sheets_output = OUTPUT_DIR / "raw_dataset_profile.csv"
    columns_output = OUTPUT_DIR / "raw_column_profile.csv"

    files_df.to_csv(files_output, index=False, encoding="utf-8-sig")
    sheets_df.to_csv(sheets_output, index=False, encoding="utf-8-sig")
    columns_df.to_csv(columns_output, index=False, encoding="utf-8-sig")

    print()
    print("=" * 70)
    print("RAW DATA PROFILING COMPLETE")
    print("=" * 70)

    print(f"Files profiled: {len(files_df)}")

    pass_count = int(
        (files_df["read_status"] == "PASS").sum()
    )

    fail_count = int(
        (files_df["read_status"] == "FAIL").sum()
    )

    skipped_count = int(
        (files_df["read_status"] == "SKIPPED").sum()
    )

    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    print(f"SKIPPED: {skipped_count}")

    print()
    print("Generated outputs:")
    print(f"  {files_output}")
    print(f"  {sheets_output}")
    print(f"  {columns_output}")

    print()
    print("Raw files were READ ONLY.")
    print("No source file was modified.")


if __name__ == "__main__":
    main()