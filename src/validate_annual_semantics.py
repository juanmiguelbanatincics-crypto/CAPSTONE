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
    / "semantic_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SOURCE FIELDS
# ============================================================

NUMERIC_COLUMNS = [
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

OTHER_OUTBOUND_COLUMNS = [
    "Qty_out_Purchase_Returns",
    "Qty_out_Sales",
    "Qty_out_Expense_Issuances",
    "Qty_out_Material_Productions",
    "Qty_out_Qty_Adjustments",
]

MOVEMENT_COLUMNS = [
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
]


# ============================================================
# HELPERS
# ============================================================

def infer_year(filename: str) -> int:
    match = re.search(
        r"(20\d{2})",
        filename,
    )

    if not match:
        raise ValueError(
            f"Could not determine year: {filename}"
        )

    return int(match.group(1))


def read_csv_safely(
    path: Path,
) -> pd.DataFrame:

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


def normalize_text(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def percent(
    numerator: int | float,
    denominator: int | float,
) -> float | None:

    if denominator == 0:
        return None

    return round(
        numerator
        / denominator
        * 100,
        4,
    )


# ============================================================
# LOAD MEDICINE PANEL
# ============================================================

def load_panel() -> pd.DataFrame:

    frames = []

    for path in sorted(
        ANNUAL_DIR.glob("*.csv")
    ):
        year = infer_year(path.name)

        dataframe = read_csv_safely(
            path
        )

        dataframe.columns = [
            str(column).strip()
            for column
            in dataframe.columns
        ]

        for column in [
            "Category1",
            "Generic_Name",
            "Brand_Name",
            "Unit",
        ]:
            dataframe[column] = normalize_text(
                dataframe[column]
            )

        for column in NUMERIC_COLUMNS:
            dataframe[column] = (
                pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )
            )

        medicines = (
            dataframe.loc[
                dataframe["Category1"]
                .fillna("")
                .str.upper()
                .eq("MEDICINES")
            ]
            .copy()
        )

        medicines["Year"] = year
        medicines["Source_File"] = (
            path.name
        )

        frames.append(
            medicines
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# NUMERIC BLANK/ZERO PROFILE
# ============================================================

def profile_numeric_semantics(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        panel.groupby("Year")
    ):
        for column in NUMERIC_COLUMNS:

            series = year_df[column]

            missing_count = int(
                series.isna().sum()
            )

            zero_count = int(
                series.eq(0).sum()
            )

            positive_count = int(
                series.gt(0).sum()
            )

            negative_count = int(
                series.lt(0).sum()
            )

            rows.append(
                {
                    "year": int(year),
                    "field": column,
                    "row_count": (
                        len(series)
                    ),
                    "missing_count": (
                        missing_count
                    ),
                    "missing_pct": percent(
                        missing_count,
                        len(series),
                    ),
                    "explicit_zero_count": (
                        zero_count
                    ),
                    "positive_count": (
                        positive_count
                    ),
                    "negative_count": (
                        negative_count
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# STOCK-ISSUANCE BLANK CONTEXT
# ============================================================

def profile_stock_issuance_blanks(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        panel.groupby("Year")
    ):

        stock = (
            year_df[
                "Qty_out_Stock_Issuances"
            ]
        )

        total_out = (
            year_df[
                "Qty_out_Total_Out"
            ]
        )

        missing_stock = stock.isna()

        other_outbound_activity = (
            year_df[
                OTHER_OUTBOUND_COLUMNS
            ]
            .fillna(0)
            .abs()
            .gt(0)
            .any(axis=1)
        )

        rows.append(
            {
                "year": int(year),

                "medicine_rows":
                    len(year_df),

                "stock_issuance_missing":
                    int(
                        missing_stock.sum()
                    ),

                "stock_issuance_explicit_zero":
                    int(
                        stock.eq(0).sum()
                    ),

                "stock_issuance_positive":
                    int(
                        stock.gt(0).sum()
                    ),

                "stock_issuance_negative":
                    int(
                        stock.lt(0).sum()
                    ),

                "missing_stock_and_total_out_missing":
                    int(
                        (
                            missing_stock
                            & total_out.isna()
                        ).sum()
                    ),

                "missing_stock_and_total_out_zero":
                    int(
                        (
                            missing_stock
                            & total_out.eq(0)
                        ).sum()
                    ),

                "missing_stock_and_total_out_nonzero":
                    int(
                        (
                            missing_stock
                            & total_out.fillna(0)
                            .ne(0)
                        ).sum()
                    ),

                "missing_stock_with_other_outbound_activity":
                    int(
                        (
                            missing_stock
                            & other_outbound_activity
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BALANCE SEMANTICS
# ============================================================

def profile_balance_semantics(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        panel.groupby("Year")
    ):

        beginning = (
            year_df[
                "Beginning_Balance"
            ]
        )

        ending = (
            year_df[
                "Ending_Balance"
            ]
        )

        any_movement = (
            year_df[
                MOVEMENT_COLUMNS
            ]
            .fillna(0)
            .abs()
            .gt(0)
            .any(axis=1)
        )

        both_missing = (
            beginning.isna()
            & ending.isna()
        )

        both_present = (
            beginning.notna()
            & ending.notna()
        )

        rows.append(
            {
                "year": int(year),

                "medicine_rows":
                    len(year_df),

                "beginning_balance_missing":
                    int(
                        beginning.isna().sum()
                    ),

                "ending_balance_missing":
                    int(
                        ending.isna().sum()
                    ),

                "both_balances_missing":
                    int(
                        both_missing.sum()
                    ),

                "both_balances_present":
                    int(
                        both_present.sum()
                    ),

                "both_balances_missing_with_any_movement":
                    int(
                        (
                            both_missing
                            & any_movement
                        ).sum()
                    ),

                "both_balances_missing_with_no_movement":
                    int(
                        (
                            both_missing
                            & ~any_movement
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BLANK-AS-ZERO STOCK FLOW TEST
# ============================================================

def test_blank_as_zero_stock_flow(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        panel.groupby("Year")
    ):

        # Beginning and ending balances must
        # remain observed. We do NOT convert
        # missing balances to zero.
        eligible = (
            year_df[
                "Beginning_Balance"
            ].notna()
            &
            year_df[
                "Ending_Balance"
            ].notna()
        )

        test_df = (
            year_df.loc[
                eligible,
                [
                    "Beginning_Balance",
                    "Qty_in_Total_In",
                    "Qty_out_Total_Out",
                    "Ending_Balance",
                ],
            ]
            .copy()
        )

        original_complete = (
            test_df[
                [
                    "Beginning_Balance",
                    "Qty_in_Total_In",
                    "Qty_out_Total_Out",
                    "Ending_Balance",
                ]
            ]
            .notna()
            .all(axis=1)
        )

        original_df = (
            test_df.loc[
                original_complete
            ]
            .copy()
        )

        original_expected = (
            original_df[
                "Beginning_Balance"
            ]
            + original_df[
                "Qty_in_Total_In"
            ]
            - original_df[
                "Qty_out_Total_Out"
            ]
        )

        original_matches = int(
            np.isclose(
                original_expected,
                original_df[
                    "Ending_Balance"
                ],
                atol=0.0001,
                rtol=0,
            ).sum()
        )

        blank_zero_df = (
            test_df.copy()
        )

        blank_zero_df[
            "Qty_in_Total_In"
        ] = (
            blank_zero_df[
                "Qty_in_Total_In"
            ]
            .fillna(0)
        )

        blank_zero_df[
            "Qty_out_Total_Out"
        ] = (
            blank_zero_df[
                "Qty_out_Total_Out"
            ]
            .fillna(0)
        )

        zero_expected = (
            blank_zero_df[
                "Beginning_Balance"
            ]
            + blank_zero_df[
                "Qty_in_Total_In"
            ]
            - blank_zero_df[
                "Qty_out_Total_Out"
            ]
        )

        blank_zero_matches = int(
            np.isclose(
                zero_expected,
                blank_zero_df[
                    "Ending_Balance"
                ],
                atol=0.0001,
                rtol=0,
            ).sum()
        )

        rows.append(
            {
                "year": int(year),

                "original_complete_rows":
                    len(original_df),

                "original_matches":
                    original_matches,

                "original_match_pct":
                    percent(
                        original_matches,
                        len(original_df),
                    ),

                "rows_with_observed_begin_and_end":
                    len(blank_zero_df),

                "blank_as_zero_matches":
                    blank_zero_matches,

                "blank_as_zero_match_pct":
                    percent(
                        blank_zero_matches,
                        len(blank_zero_df),
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MEDICINE IDENTITY STABILITY
# ============================================================

def profile_identity_stability(
    panel: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    grouped = (
        panel.groupby(
            "Brand_Name",
            dropna=False,
        )
        .agg(
            years_observed=(
                "Year",
                "nunique",
            ),
            generic_name_count=(
                "Generic_Name",
                lambda x:
                    x.dropna().nunique(),
            ),
            unit_count=(
                "Unit",
                lambda x:
                    x.dropna().nunique(),
            ),
            category_count=(
                "Category1",
                lambda x:
                    x.dropna().nunique(),
            ),
        )
        .reset_index()
    )

    grouped[
        "identity_stable"
    ] = (
        grouped[
            "years_observed"
        ].eq(5)
        &
        grouped[
            "generic_name_count"
        ].le(1)
        &
        grouped[
            "unit_count"
        ].le(1)
        &
        grouped[
            "category_count"
        ].le(1)
    )

    conflicts = grouped.loc[
        ~grouped[
            "identity_stable"
        ]
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "unique_brand_names":
                    len(grouped),

                "observed_all_5_years":
                    int(
                        grouped[
                            "years_observed"
                        ]
                        .eq(5)
                        .sum()
                    ),

                "stable_generic_name":
                    int(
                        grouped[
                            "generic_name_count"
                        ]
                        .le(1)
                        .sum()
                    ),

                "stable_unit":
                    int(
                        grouped[
                            "unit_count"
                        ]
                        .le(1)
                        .sum()
                    ),

                "fully_stable_identity":
                    int(
                        grouped[
                            "identity_stable"
                        ]
                        .sum()
                    ),

                "identity_conflicts":
                    len(conflicts),
            }
        ]
    )

    return summary, conflicts


# ============================================================
# COST PROFILE
# ============================================================

def profile_cost_semantics(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for year, year_df in (
        panel.groupby("Year")
    ):

        cost = year_df[
            "Average_cost"
        ]

        issuance = (
            year_df[
                "Qty_out_Stock_Issuances"
            ]
            .fillna(0)
        )

        invalid_cost = (
            cost.isna()
            | cost.le(0)
        )

        rows.append(
            {
                "year": int(year),

                "cost_missing":
                    int(
                        cost.isna().sum()
                    ),

                "cost_zero":
                    int(
                        cost.eq(0).sum()
                    ),

                "cost_negative":
                    int(
                        cost.lt(0).sum()
                    ),

                "cost_positive":
                    int(
                        cost.gt(0).sum()
                    ),

                "positive_issuance_with_invalid_cost":
                    int(
                        (
                            issuance.gt(0)
                            & invalid_cost
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 78)
    print("CDH INVENTORY DSS")
    print("ANNUAL DATA SEMANTIC VALIDATION")
    print("=" * 78)
    print()

    panel = load_panel()

    print(
        "Medicine-year records loaded:",
        f"{len(panel):,}",
    )

    print(
        "Unique Brand_Name values:",
        f"{panel['Brand_Name'].nunique():,}",
    )

    print()

    numeric_profile = (
        profile_numeric_semantics(
            panel
        )
    )

    stock_blank_profile = (
        profile_stock_issuance_blanks(
            panel
        )
    )

    balance_profile = (
        profile_balance_semantics(
            panel
        )
    )

    flow_profile = (
        test_blank_as_zero_stock_flow(
            panel
        )
    )

    identity_summary, identity_conflicts = (
        profile_identity_stability(
            panel
        )
    )

    cost_profile = (
        profile_cost_semantics(
            panel
        )
    )

    # ========================================================
    # PRINT KEY RESULTS
    # ========================================================

    print("-" * 78)
    print("STOCK ISSUANCE BLANK CONTEXT")
    print("-" * 78)

    print(
        stock_blank_profile
        .to_string(index=False)
    )

    print()
    print("-" * 78)
    print("BALANCE AVAILABILITY")
    print("-" * 78)

    print(
        balance_profile
        .to_string(index=False)
    )

    print()
    print("-" * 78)
    print("BLANK-AS-ZERO STOCK FLOW TEST")
    print("-" * 78)

    print(
        flow_profile
        .to_string(index=False)
    )

    print()
    print("-" * 78)
    print("MEDICINE IDENTITY STABILITY")
    print("-" * 78)

    print(
        identity_summary
        .to_string(index=False)
    )

    if len(
        identity_conflicts
    ) > 0:

        print()
        print(
            "Identity conflicts detected:",
            len(identity_conflicts),
        )

    else:
        print()
        print(
            "No medicine identity "
            "conflicts detected."
        )

    print()
    print("-" * 78)
    print("COST SEMANTICS")
    print("-" * 78)

    print(
        cost_profile
        .to_string(index=False)
    )

    # ========================================================
    # SAVE
    # ========================================================

    numeric_profile.to_csv(
        OUTPUT_DIR
        / "numeric_field_semantics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stock_blank_profile.to_csv(
        OUTPUT_DIR
        / "stock_issuance_blank_context.csv",
        index=False,
        encoding="utf-8-sig",
    )

    balance_profile.to_csv(
        OUTPUT_DIR
        / "balance_semantics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    flow_profile.to_csv(
        OUTPUT_DIR
        / "blank_as_zero_stockflow_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    identity_summary.to_csv(
        OUTPUT_DIR
        / "medicine_identity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    identity_conflicts.to_csv(
        OUTPUT_DIR
        / "medicine_identity_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cost_profile.to_csv(
        OUTPUT_DIR
        / "cost_semantics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("SEMANTIC VALIDATION COMPLETE")
    print("=" * 78)

    print()
    print("Generated outputs:")

    for file in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            " ",
            file.relative_to(
                PROJECT_ROOT
            ),
        )

    print()
    print(
        "No raw source file was modified."
    )

    print(
        "No missing value has yet been "
        "converted to zero in a validated "
        "production dataset."
    )


if __name__ == "__main__":
    main()
    