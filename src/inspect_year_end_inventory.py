from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YEAR_END_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "year_end_inventory"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "supporting_source_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def infer_year(filename: str) -> int | None:
    match = re.search(
        r"(20\d{2})",
        filename,
    )

    if match:
        return int(match.group(1))

    return None


def clean_value(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() == "nan":
        return ""

    return text


def compact_row(
    dataframe: pd.DataFrame,
    row_number: int,
) -> str:

    row = dataframe.iloc[row_number]

    values = []

    for column_index, value in enumerate(row):
        text = clean_value(value)

        if text:
            if len(text) > 80:
                text = text[:77] + "..."

            values.append(
                f"C{column_index + 1}={text}"
            )

    if not values:
        return "<EMPTY>"

    return " | ".join(values)


def header_score(row: pd.Series) -> int:
    """
    Score a row according to terms likely to appear
    in an inventory-table header.
    """

    keywords = [
        "item",
        "generic",
        "brand",
        "description",
        "medicine",
        "unit",
        "balance",
        "qty",
        "quantity",
        "cost",
        "price",
        "inventory",
        "beginning",
        "ending",
        "stock",
    ]

    text = " ".join(
        clean_value(value).lower()
        for value in row
        if clean_value(value)
    )

    return sum(
        1
        for keyword in keywords
        if keyword in text
    )


def find_keyword_cells(
    dataframe: pd.DataFrame,
    keywords: list[str],
) -> list[dict]:

    matches = []

    for row_index in range(
        len(dataframe)
    ):
        for column_index in range(
            len(dataframe.columns)
        ):

            value = clean_value(
                dataframe.iat[
                    row_index,
                    column_index,
                ]
            )

            if not value:
                continue

            lower = value.lower()

            for keyword in keywords:
                if keyword.lower() in lower:

                    matches.append(
                        {
                            "row_excel":
                                row_index + 1,
                            "column_number":
                                column_index + 1,
                            "keyword":
                                keyword,
                            "cell_value":
                                value,
                        }
                    )

                    break

    return matches


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 82)
    print("CDH INVENTORY DSS")
    print("YEAR-END INVENTORY SUPPORTING SOURCE INSPECTION")
    print("=" * 82)
    print()

    workbooks = sorted(
        YEAR_END_DIR.glob("*.xlsx")
    )

    if not workbooks:
        raise FileNotFoundError(
            f"No XLSX files found under: {YEAR_END_DIR}"
        )

    print(
        "Year-end workbooks detected:",
        len(workbooks),
    )

    print()

    workbook_summary = []
    keyword_results = []
    header_candidates = []

    report_lines = []

    for workbook_path in workbooks:

        year = infer_year(
            workbook_path.name
        )

        print("-" * 82)
        print(
            f"WORKBOOK: {workbook_path.name}"
        )
        print(
            f"YEAR: {year}"
        )

        excel_file = pd.ExcelFile(
            workbook_path,
            engine="openpyxl",
        )

        print(
            "Sheets:",
            excel_file.sheet_names,
        )

        for sheet_name in (
            excel_file.sheet_names
        ):

            raw = pd.read_excel(
                workbook_path,
                sheet_name=sheet_name,
                header=None,
                engine="openpyxl",
            )

            # Remove completely blank rows/columns only
            # for structural reporting.
            raw_nonempty = (
                raw
                .dropna(
                    axis=0,
                    how="all",
                )
                .dropna(
                    axis=1,
                    how="all",
                )
            )

            print()
            print(
                f"Sheet: {sheet_name}"
            )
            print(
                "Raw dimensions:",
                f"{len(raw):,} rows x "
                f"{len(raw.columns):,} columns",
            )

            print(
                "Non-empty dimensions:",
                f"{len(raw_nonempty):,} rows x "
                f"{len(raw_nonempty.columns):,} columns",
            )

            workbook_summary.append(
                {
                    "year": year,
                    "source_file":
                        workbook_path.name,
                    "sheet_name":
                        sheet_name,
                    "raw_rows":
                        len(raw),
                    "raw_columns":
                        len(raw.columns),
                    "nonempty_rows":
                        len(raw_nonempty),
                    "nonempty_columns":
                        len(
                            raw_nonempty.columns
                        ),
                }
            )

            # ------------------------------------------------
            # HEADER CANDIDATES
            # ------------------------------------------------

            scores = []

            inspect_limit = min(
                40,
                len(raw),
            )

            for row_index in range(
                inspect_limit
            ):

                score = header_score(
                    raw.iloc[row_index]
                )

                scores.append(
                    (
                        score,
                        row_index,
                    )
                )

            scores = sorted(
                scores,
                reverse=True,
            )

            top_candidates = [
                item
                for item in scores
                if item[0] > 0
            ][:5]

            print()
            print(
                "Top probable header rows:"
            )

            if not top_candidates:
                print(
                    "  No strong candidate detected."
                )

            for score, row_index in (
                top_candidates
            ):

                content = compact_row(
                    raw,
                    row_index,
                )

                print(
                    f"  Excel row "
                    f"{row_index + 1}: "
                    f"score={score}"
                )
                print(
                    f"    {content}"
                )

                header_candidates.append(
                    {
                        "year": year,
                        "source_file":
                            workbook_path.name,
                        "sheet_name":
                            sheet_name,
                        "excel_row":
                            row_index + 1,
                        "score":
                            score,
                        "row_content":
                            content,
                    }
                )

            # ------------------------------------------------
            # FIRST 20 ROWS
            # ------------------------------------------------

            print()
            print(
                "First 20 physical rows:"
            )

            report_lines.append(
                "=" * 100
            )

            report_lines.append(
                f"{workbook_path.name} | "
                f"{sheet_name}"
            )

            report_lines.append(
                "=" * 100
            )

            preview_limit = min(
                20,
                len(raw),
            )

            for row_index in range(
                preview_limit
            ):

                content = compact_row(
                    raw,
                    row_index,
                )

                line = (
                    f"Excel row "
                    f"{row_index + 1}: "
                    f"{content}"
                )

                print(
                    " ",
                    line,
                )

                report_lines.append(
                    line
                )

            report_lines.append("")

            # ------------------------------------------------
            # SEARCH IMPORTANT TERMS
            # ------------------------------------------------

            search_keywords = [
                "medicine",
                "medicines",
                "generic",
                "brand",
                "description",
                "unit",
                "balance",
                "cost",
                "price",
                "quantity",
                "qty",
            ]

            matches = find_keyword_cells(
                raw,
                search_keywords,
            )

            for result in matches:
                result.update(
                    {
                        "year": year,
                        "source_file":
                            workbook_path.name,
                        "sheet_name":
                            sheet_name,
                    }
                )

            keyword_results.extend(
                matches
            )

            print()
            print(
                "Keyword cells detected:",
                len(matches),
            )

            for result in matches[:20]:

                print(
                    f"  Row "
                    f"{result['row_excel']}, "
                    f"Column "
                    f"{result['column_number']}: "
                    f"{result['cell_value']}"
                )

        print()

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    summary_df = pd.DataFrame(
        workbook_summary
    )

    candidate_df = pd.DataFrame(
        header_candidates
    )

    keyword_df = pd.DataFrame(
        keyword_results
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "year_end_workbook_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidate_df.to_csv(
        OUTPUT_DIR
        / "year_end_header_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    keyword_df.to_csv(
        OUTPUT_DIR
        / "year_end_keyword_cells.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report_path = (
        OUTPUT_DIR
        / "year_end_first20rows_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 82)
    print(
        "YEAR-END SOURCE INSPECTION COMPLETE"
    )
    print("=" * 82)

    print()
    print(
        "Generated outputs:"
    )

    for file in sorted(
        OUTPUT_DIR.glob("*")
    ):

        print(
            " ",
            file.relative_to(
                PROJECT_ROOT
            ),
        )

    print()
    print(
        "No year-end workbook was modified."
    )

    print(
        "No value from the supporting files "
        "has yet been used to replace annual "
        "inventory data."
    )


if __name__ == "__main__":
    main()
    