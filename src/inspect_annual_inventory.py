from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNUAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "annual_inventory"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "data_understanding"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED ANNUAL REPORT STRUCTURE
# ============================================================

EXPECTED_COLUMNS = [
    "Category1",
    "Generic_Name",
    "Brand_Name",
    "Unit",
    "Beginning_Balance",
    "Qty_in_Deliveries",
    "Qty_in_Credit_notes",
    "Qty_in_Stock_Issuances",
    "Qty_in_Material_Productions",
    "Qty_in_Qty_Adjustments",
    "Qty_in_Total_In",
    "Qty_out_Purchase_Returns",
    "Qty_out_Sales",
    "Qty_out_Stock_Issuances",
    "Qty_out_Expense_Issuances",
    "Qty_out_Material_Productions",
    "Qty_out_Qty_Adjustments",
    "Qty_out_Total_Out",
    "Ending_Balance",
    "Average_cost",
    "Textbox48",
    "Textbox1",
    "Textbox2",
]


# ============================================================
# HELPERS
# ============================================================

def infer_year(filename: str) -> int:
    match = re.search(r"(20\d{2})", filename)

    if not match:
        raise ValueError(
            f"Could not infer year from filename: {filename}"
        )

    return int(match.group(1))


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError as error:
            last_error = error

    raise last_error


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
    )


def to_numeric(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:

    result = dataframe.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def percent(
    numerator: float,
    denominator: float,
) -> float | None:

    if denominator == 0:
        return None

    return round(
        (numerator / denominator) * 100,
        4,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main() -> None:

    print("=" * 78)
    print("CDH INVENTORY DSS")
    print("ANNUAL BIZBOX INVENTORY — DATA UNDERSTANDING")
    print("=" * 78)
    print()

    annual_files = sorted(
        ANNUAL_DIR.glob("*.csv")
    )

    if not annual_files:
        raise FileNotFoundError(
            f"No annual CSV files found in {ANNUAL_DIR}"
        )

    print(
        f"Annual inventory files detected: "
        f"{len(annual_files)}"
    )
    print()

    file_summaries = []
    medicine_year_frames = []
    category_summaries = []
    stock_flow_summaries = []
    quality_summaries = []
    movement_summaries = []

    reference_columns = None

    numeric_columns = [
        "Beginning_Balance",
        "Qty_in_Deliveries",
        "Qty_in_Credit_notes",
        "Qty_in_Stock_Issuances",
        "Qty_in_Material_Productions",
        "Qty_in_Qty_Adjustments",
        "Qty_in_Total_In",
        "Qty_out_Purchase_Returns",
        "Qty_out_Sales",
        "Qty_out_Stock_Issuances",
        "Qty_out_Expense_Issuances",
        "Qty_out_Material_Productions",
        "Qty_out_Qty_Adjustments",
        "Qty_out_Total_Out",
        "Ending_Balance",
        "Average_cost",
    ]

    for path in annual_files:

        year = infer_year(path.name)

        print("-" * 78)
        print(f"FILE: {path.name}")
        print(f"YEAR: {year}")

        dataframe = read_csv_safely(path)

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        actual_columns = list(
            dataframe.columns
        )

        # ----------------------------------------------------
        # STRUCTURE CHECK
        # ----------------------------------------------------

        if reference_columns is None:
            reference_columns = actual_columns

        same_as_reference = (
            actual_columns == reference_columns
        )

        exact_expected_structure = (
            actual_columns == EXPECTED_COLUMNS
        )

        missing_expected = [
            column
            for column in EXPECTED_COLUMNS
            if column not in actual_columns
        ]

        unexpected_columns = [
            column
            for column in actual_columns
            if column not in EXPECTED_COLUMNS
        ]

        print(
            "Rows:",
            f"{len(dataframe):,}",
        )

        print(
            "Columns:",
            len(dataframe.columns),
        )

        print(
            "Same column order as first annual file:",
            same_as_reference,
        )

        print(
            "Matches expected Bizbox structure:",
            exact_expected_structure,
        )

        if missing_expected:
            print(
                "Missing expected columns:",
                missing_expected,
            )

        if unexpected_columns:
            print(
                "Unexpected columns:",
                unexpected_columns,
            )

        # ----------------------------------------------------
        # STANDARDIZE TEXT FOR ANALYSIS
        # ----------------------------------------------------

        for text_column in [
            "Category1",
            "Generic_Name",
            "Brand_Name",
            "Unit",
        ]:
            if text_column in dataframe.columns:
                dataframe[text_column] = normalize_text(
                    dataframe[text_column]
                )

        dataframe = to_numeric(
            dataframe,
            numeric_columns,
        )

        # ----------------------------------------------------
        # CATEGORY PROFILE
        # ----------------------------------------------------

        category_counts = (
            dataframe["Category1"]
            .fillna("<MISSING>")
            .value_counts(dropna=False)
        )

        print()
        print("Top Category1 values:")

        for category, count in (
            category_counts
            .head(10)
            .items()
        ):
            print(
                f"  {category}: {count:,}"
            )

        for category, count in (
            category_counts.items()
        ):
            category_summaries.append(
                {
                    "year": year,
                    "source_file": path.name,
                    "category": category,
                    "row_count": int(count),
                }
            )

        # ----------------------------------------------------
        # MEDICINES ONLY
        # ----------------------------------------------------

        medicine_mask = (
            dataframe["Category1"]
            .fillna("")
            .str.upper()
            .eq("MEDICINES")
        )

        medicines = (
            dataframe.loc[medicine_mask]
            .copy()
        )

        medicines["Year"] = year
        medicines["Source_File"] = path.name

        medicine_rows = len(medicines)

        unique_brand = (
            medicines["Brand_Name"]
            .dropna()
            .nunique()
        )

        unique_generic = (
            medicines["Generic_Name"]
            .dropna()
            .nunique()
        )

        duplicate_brand_rows = int(
            medicines["Brand_Name"]
            .duplicated(keep=False)
            .sum()
        )

        exact_duplicate_rows = int(
            medicines
            .drop(
                columns=[
                    "Year",
                    "Source_File",
                ],
                errors="ignore",
            )
            .duplicated()
            .sum()
        )

        print()
        print("MEDICINES subset:")
        print(
            "  Medicine rows:",
            f"{medicine_rows:,}",
        )
        print(
            "  Unique Brand_Name:",
            f"{unique_brand:,}",
        )
        print(
            "  Unique Generic_Name:",
            f"{unique_generic:,}",
        )
        print(
            "  Rows with duplicated Brand_Name:",
            f"{duplicate_brand_rows:,}",
        )
        print(
            "  Exact duplicate rows:",
            f"{exact_duplicate_rows:,}",
        )

        # ----------------------------------------------------
        # STOCK FLOW RECONCILIATION
        # ----------------------------------------------------

        required_flow_columns = [
            "Beginning_Balance",
            "Qty_in_Total_In",
            "Qty_out_Total_Out",
            "Ending_Balance",
        ]

        flow_complete = (
            medicines[
                required_flow_columns
            ]
            .notna()
            .all(axis=1)
        )

        flow_df = (
            medicines.loc[
                flow_complete
            ]
            .copy()
        )

        flow_df[
            "Expected_Ending_Balance"
        ] = (
            flow_df["Beginning_Balance"]
            + flow_df["Qty_in_Total_In"]
            - flow_df["Qty_out_Total_Out"]
        )

        flow_df[
            "Stock_Flow_Difference"
        ] = (
            flow_df["Ending_Balance"]
            - flow_df[
                "Expected_Ending_Balance"
            ]
        )

        flow_df[
            "Stock_Flow_Reconciled"
        ] = np.isclose(
            flow_df["Ending_Balance"],
            flow_df[
                "Expected_Ending_Balance"
            ],
            atol=0.0001,
            rtol=0,
        )

        reconciled_count = int(
            flow_df[
                "Stock_Flow_Reconciled"
            ].sum()
        )

        flow_total = len(flow_df)

        flow_rate = percent(
            reconciled_count,
            flow_total,
        )

        print()
        print("Stock-flow validation:")
        print(
            "  Complete rows checked:",
            f"{flow_total:,}",
        )
        print(
            "  Reconciled:",
            f"{reconciled_count:,}",
        )
        print(
            "  Reconciliation rate:",
            (
                f"{flow_rate}%"
                if flow_rate is not None
                else "N/A"
            ),
        )

        stock_flow_summaries.append(
            {
                "year": year,
                "source_file": path.name,
                "rows_checked": flow_total,
                "rows_reconciled": reconciled_count,
                "reconciliation_rate_pct": (
                    flow_rate
                ),
                "max_abs_difference": (
                    float(
                        flow_df[
                            "Stock_Flow_Difference"
                        ]
                        .abs()
                        .max()
                    )
                    if flow_total > 0
                    else None
                ),
            }
        )

        # ----------------------------------------------------
        # MOVEMENT TOTALS
        # ----------------------------------------------------

        movement_fields = [
            "Qty_in_Deliveries",
            "Qty_in_Total_In",
            "Qty_out_Stock_Issuances",
            "Qty_out_Expense_Issuances",
            "Qty_out_Qty_Adjustments",
            "Qty_out_Purchase_Returns",
            "Qty_out_Sales",
            "Qty_out_Material_Productions",
            "Qty_out_Total_Out",
        ]

        movement_row = {
            "year": year,
            "source_file": path.name,
        }

        for field in movement_fields:
            if field in medicines.columns:

                total_value = float(
                    medicines[field]
                    .fillna(0)
                    .sum()
                )

                active_items = int(
                    (
                        medicines[field]
                        .fillna(0)
                        != 0
                    )
                    .sum()
                )

                movement_row[
                    f"{field}_total"
                ] = total_value

                movement_row[
                    f"{field}_active_rows"
                ] = active_items

        movement_summaries.append(
            movement_row
        )

        print()
        print("Selected MEDICINES movement totals:")

        for field in [
            "Qty_in_Deliveries",
            "Qty_out_Stock_Issuances",
            "Qty_out_Expense_Issuances",
            "Qty_out_Qty_Adjustments",
            "Qty_out_Total_Out",
        ]:

            if field in medicines.columns:

                total_value = (
                    medicines[field]
                    .fillna(0)
                    .sum()
                )

                active_count = int(
                    (
                        medicines[field]
                        .fillna(0)
                        != 0
                    )
                    .sum()
                )

                print(
                    f"  {field}: "
                    f"{total_value:,.2f} "
                    f"({active_count:,} active rows)"
                )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        negative_begin = int(
            (
                medicines["Beginning_Balance"]
                < 0
            )
            .sum()
        )

        negative_end = int(
            (
                medicines["Ending_Balance"]
                < 0
            )
            .sum()
        )

        negative_cost = int(
            (
                medicines["Average_cost"]
                < 0
            )
            .sum()
        )

        zero_cost = int(
            (
                medicines["Average_cost"]
                == 0
            )
            .sum()
        )

        missing_brand = int(
            medicines[
                "Brand_Name"
            ]
            .isna()
            .sum()
        )

        missing_generic = int(
            medicines[
                "Generic_Name"
            ]
            .isna()
            .sum()
        )

        missing_stock_issuance = int(
            medicines[
                "Qty_out_Stock_Issuances"
            ]
            .isna()
            .sum()
        )

        quality_summaries.append(
            {
                "year": year,
                "source_file": path.name,
                "medicine_rows": medicine_rows,
                "unique_brand_names": (
                    unique_brand
                ),
                "unique_generic_names": (
                    unique_generic
                ),
                "duplicate_brand_rows": (
                    duplicate_brand_rows
                ),
                "exact_duplicate_rows": (
                    exact_duplicate_rows
                ),
                "missing_brand_name": (
                    missing_brand
                ),
                "missing_generic_name": (
                    missing_generic
                ),
                "missing_stock_issuance": (
                    missing_stock_issuance
                ),
                "negative_beginning_balance": (
                    negative_begin
                ),
                "negative_ending_balance": (
                    negative_end
                ),
                "negative_average_cost": (
                    negative_cost
                ),
                "zero_average_cost": (
                    zero_cost
                ),
            }
        )

        print()
        print("Data-quality indicators:")
        print(
            "  Missing Brand_Name:",
            f"{missing_brand:,}",
        )
        print(
            "  Missing Generic_Name:",
            f"{missing_generic:,}",
        )
        print(
            "  Missing Stock Issuance:",
            f"{missing_stock_issuance:,}",
        )
        print(
            "  Negative Beginning Balance:",
            f"{negative_begin:,}",
        )
        print(
            "  Negative Ending Balance:",
            f"{negative_end:,}",
        )
        print(
            "  Negative Average Cost:",
            f"{negative_cost:,}",
        )
        print(
            "  Zero Average Cost:",
            f"{zero_cost:,}",
        )

        file_summaries.append(
            {
                "year": year,
                "source_file": path.name,
                "all_rows": len(dataframe),
                "all_columns": len(
                    dataframe.columns
                ),
                "same_structure_as_first_file": (
                    same_as_reference
                ),
                "exact_expected_structure": (
                    exact_expected_structure
                ),
                "medicine_rows": medicine_rows,
                "unique_brand_names": unique_brand,
                "unique_generic_names": (
                    unique_generic
                ),
            }
        )

        medicine_year_frames.append(
            medicines
        )

    # ========================================================
    # COMBINE MEDICINE-YEAR PANEL
    # ========================================================

    panel = pd.concat(
        medicine_year_frames,
        ignore_index=True,
    )

    print()
    print("=" * 78)
    print("CROSS-YEAR MEDICINE PANEL CHECK")
    print("=" * 78)

    print(
        "Medicine-year rows:",
        f"{len(panel):,}",
    )

    print(
        "Years:",
        sorted(
            panel["Year"]
            .unique()
            .tolist()
        ),
    )

    # --------------------------------------------------------
    # BRAND NAME STABILITY
    # --------------------------------------------------------

    brand_sets = {}

    for year, year_df in (
        panel.groupby("Year")
    ):
        brand_sets[int(year)] = set(
            year_df["Brand_Name"]
            .dropna()
            .astype(str)
            .str.strip()
        )

    all_years = sorted(
        brand_sets.keys()
    )

    if all_years:
        common_brands = set.intersection(
            *[
                brand_sets[year]
                for year in all_years
            ]
        )

        union_brands = set.union(
            *[
                brand_sets[year]
                for year in all_years
            ]
        )

        print()
        print(
            "Unique Brand_Name values "
            "common to ALL years:",
            f"{len(common_brands):,}",
        )

        print(
            "Unique Brand_Name values "
            "across ANY year:",
            f"{len(union_brands):,}",
        )

        if len(union_brands) > 0:
            common_pct = (
                len(common_brands)
                / len(union_brands)
                * 100
            )
        else:
            common_pct = 0

        print(
            "Common-across-all-years percentage:",
            f"{common_pct:.2f}%",
        )

    continuity_rows = []

    for current_year, next_year in zip(
        all_years[:-1],
        all_years[1:],
    ):

        current_df = (
            panel.loc[
                panel["Year"]
                .eq(current_year),
                [
                    "Brand_Name",
                    "Ending_Balance",
                ],
            ]
            .dropna(
                subset=["Brand_Name"]
            )
            .copy()
        )

        next_df = (
            panel.loc[
                panel["Year"]
                .eq(next_year),
                [
                    "Brand_Name",
                    "Beginning_Balance",
                ],
            ]
            .dropna(
                subset=["Brand_Name"]
            )
            .copy()
        )

        current_df = current_df.rename(
            columns={
                "Ending_Balance":
                    "Prior_Ending_Balance"
            }
        )

        next_df = next_df.rename(
            columns={
                "Beginning_Balance":
                    "Next_Beginning_Balance"
            }
        )

        merged = current_df.merge(
            next_df,
            on="Brand_Name",
            how="inner",
        )

        complete = merged[
            [
                "Prior_Ending_Balance",
                "Next_Beginning_Balance",
            ]
        ].notna().all(axis=1)

        matched = merged.loc[
            complete
        ].copy()

        if len(matched) > 0:

            matched[
                "Balance_Continues"
            ] = np.isclose(
                matched[
                    "Prior_Ending_Balance"
                ],
                matched[
                    "Next_Beginning_Balance"
                ],
                atol=0.0001,
                rtol=0,
            )

            match_count = int(
                matched[
                    "Balance_Continues"
                ].sum()
            )

            match_rate = percent(
                match_count,
                len(matched),
            )

        else:
            match_count = 0
            match_rate = None

        continuity_rows.append(
            {
                "from_year": current_year,
                "to_year": next_year,
                "medicine_pairs_checked": (
                    len(matched)
                ),
                "balance_matches": (
                    match_count
                ),
                "balance_continuity_pct": (
                    match_rate
                ),
            }
        )

        print()
        print(
            f"{current_year} Ending Balance "
            f"→ {next_year} Beginning Balance:"
        )

        print(
            "  Comparable medicines:",
            f"{len(matched):,}",
        )

        print(
            "  Exact/near-exact matches:",
            f"{match_count:,}",
        )

        print(
            "  Continuity rate:",
            (
                f"{match_rate}%"
                if match_rate is not None
                else "N/A"
            ),
        )

    # ========================================================
    # DEMAND TARGET DIAGNOSTIC
    # ========================================================

    print()
    print("=" * 78)
    print("DEMAND TARGET DIAGNOSTIC")
    print("=" * 78)

    stock_issuance_total = float(
        panel[
            "Qty_out_Stock_Issuances"
        ]
        .fillna(0)
        .sum()
    )

    total_out_total = float(
        panel[
            "Qty_out_Total_Out"
        ]
        .fillna(0)
        .sum()
    )

    expense_total = float(
        panel[
            "Qty_out_Expense_Issuances"
        ]
        .fillna(0)
        .sum()
    )

    adjustment_total = float(
        panel[
            "Qty_out_Qty_Adjustments"
        ]
        .fillna(0)
        .sum()
    )

    print(
        "Total Qty_out_Stock_Issuances:",
        f"{stock_issuance_total:,.2f}",
    )

    print(
        "Total Qty_out_Total_Out:",
        f"{total_out_total:,.2f}",
    )

    print(
        "Total Qty_out_Expense_Issuances:",
        f"{expense_total:,.2f}",
    )

    print(
        "Total Qty_out_Qty_Adjustments:",
        f"{adjustment_total:,.2f}",
    )

    print()
    print(
        "Interpretation rule for next step:"
    )

    print(
        "  Qty_out_Stock_Issuances is the "
        "candidate observed annual issuance/"
        "consumption proxy."
    )

    print(
        "  Qty_out_Total_Out will NOT "
        "automatically be treated as demand "
        "because it contains other outbound "
        "movement categories."
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    file_summary_df = pd.DataFrame(
        file_summaries
    )

    category_summary_df = pd.DataFrame(
        category_summaries
    )

    flow_summary_df = pd.DataFrame(
        stock_flow_summaries
    )

    quality_summary_df = pd.DataFrame(
        quality_summaries
    )

    movement_summary_df = pd.DataFrame(
        movement_summaries
    )

    continuity_df = pd.DataFrame(
        continuity_rows
    )

    # Do NOT call this validated yet.
    # It is still an understanding-stage extract.
    panel_output = (
        OUTPUT_DIR
        / "annual_medicines_understanding.csv"
    )

    file_summary_df.to_csv(
        OUTPUT_DIR
        / "annual_file_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    category_summary_df.to_csv(
        OUTPUT_DIR
        / "annual_category_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    flow_summary_df.to_csv(
        OUTPUT_DIR
        / "annual_stockflow_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    quality_summary_df.to_csv(
        OUTPUT_DIR
        / "annual_quality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    movement_summary_df.to_csv(
        OUTPUT_DIR
        / "annual_movement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    continuity_df.to_csv(
        OUTPUT_DIR
        / "annual_balance_continuity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    panel.to_csv(
        panel_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("ANNUAL DATA UNDERSTANDING COMPLETE")
    print("=" * 78)

    print()
    print("Generated files:")

    for output_file in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            " ",
            output_file.relative_to(
                PROJECT_ROOT
            ),
        )

    print()
    print(
        "No raw source file was modified."
    )

    print(
        "The combined medicine table is "
        "still a DATA-UNDERSTANDING artifact, "
        "not yet the final validated panel."
    )


if __name__ == "__main__":
    main()