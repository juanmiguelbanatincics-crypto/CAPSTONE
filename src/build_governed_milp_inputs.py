from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

try:
    from .runtime_context import (
        load_runtime_context_from_env,
    )
except ImportError:
    from runtime_context import (
        load_runtime_context_from_env,
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

SOURCE_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_optimizer_inputs"
)

MILP_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_inputs"
)

GOVERNANCE_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_milp_governance"
)

MILP_INPUT_VERSION = (
    f"MILP_INPUTS_{TARGET_YEAR}_"
    "GOVERNED_V1"
)

OUTPUT_DIR = (
    RUNTIME_CONTEXT.processed_dir
    / "planning"
    / (
        f"milp_inputs_{TARGET_YEAR}_"
        "governed_v1"
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INTEGER_TOLERANCE = 1e-9

SCENARIOS = {
    "POINT": {
        "source_requirement":
            "Planning_Requirement_Point",
        "rounding":
            "CEILING_TO_WHOLE_UNIT_IF_NEEDED",
    },
    "P80_FLOORED_CEIL": {
        "source_requirement":
            "Planning_Requirement_P80_Floored",
        "rounding":
            "CEILING_TO_WHOLE_UNIT",
    },
    "P90_FLOORED_CEIL": {
        "source_requirement":
            "Planning_Requirement_P90_Floored",
        "rounding":
            "CEILING_TO_WHOLE_UNIT",
    },
}


def is_integer_like(
    values: np.ndarray,
) -> np.ndarray:
    return (
        np.abs(
            values
            -
            np.round(
                values
            )
        )
        <=
        INTEGER_TOLERANCE
    )


def ceiling_whole_units(
    values: np.ndarray,
) -> np.ndarray:
    return np.ceil(
        values
        -
        INTEGER_TOLERANCE
    ).astype(
        np.int64
    )


def load_inputs() -> pd.DataFrame:
    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        dataframe = connection.execute(
            f"""
            SELECT *
            FROM {SOURCE_TABLE}
            ORDER BY Medicine_Key
            """
        ).df()

    finally:
        connection.close()

    return dataframe


def validate_source(
    dataframe: pd.DataFrame,
) -> None:
    if dataframe.empty:
        raise ValueError(
            "Governed optimizer input "
            "population is empty."
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate Medicine_Key."
        )

    if not (
        dataframe[
            "Target_Year"
        ]
        .eq(
            TARGET_YEAR
        )
        .all()
    ):
        raise ValueError(
            "Unexpected planning target year."
        )

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    if not (
        dataframe.loc[
            eligible,
            "ABC_AsOf",
        ]
        .isin(
            [
                "A",
                "B",
                "C",
            ]
        )
        .all()
    ):
        raise ValueError(
            "Non-A/B/C row entered "
            "strict priority population."
        )

    if (
        dataframe.loc[
            eligible,
            "Planning_Cost",
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Eligible row has missing cost."
        )

    if (
        dataframe.loc[
            eligible,
            "Planning_Cost",
        ]
        .le(0)
        .any()
    ):
        raise ValueError(
            "Eligible row has "
            "non-positive cost."
        )

    if (
        dataframe.loc[
            eligible,
            "Planning_Stock_Qty",
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "Eligible row has missing stock."
        )


def build_milp_inputs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    eligible = (
        result[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # STOCK: NEVER SILENTLY ROUND FRACTIONAL SOURCE STOCK
    # --------------------------------------------------------

    eligible_stock = (
        result.loc[
            eligible,
            "Planning_Stock_Qty",
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not (
        is_integer_like(
            eligible_stock
        )
        .all()
    ):
        raise ValueError(
            "Eligible stock contains "
            "fractional quantities. "
            "Current inventory evidence "
            "requires review before MILP."
        )

    result[
        "MILP_Stock_Qty"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    result.loc[
        eligible,
        "MILP_Stock_Qty",
    ] = (
        np.round(
            eligible_stock
        )
        .astype(
            np.int64
        )
    )

    # --------------------------------------------------------
    # WHOLE-UNIT PROCUREMENT REQUIREMENTS
    # --------------------------------------------------------

    for scenario, config in (
        SCENARIOS.items()
    ):
        source_column = (
            config[
                "source_requirement"
            ]
        )

        requirement_column = (
            f"MILP_Requirement_{scenario}"
        )

        net_column = (
            f"MILP_Net_Requirement_{scenario}"
        )

        value_column = (
            f"MILP_Full_Funding_Value_{scenario}"
        )

        increment_column = (
            f"MILP_Rounding_Increment_{scenario}"
        )

        source_values = pd.to_numeric(
            result[
                source_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        if not np.isfinite(
            source_values
        ).all():
            raise ValueError(
                f"{source_column} contains "
                "missing/non-finite values."
            )

        if (
            source_values
            < 0
        ).any():
            raise ValueError(
                f"{source_column} contains "
                "negative requirements."
            )

        # Statistical/planning values remain untouched.
        # Whole-unit conversion occurs only here,
        # at the procurement decision boundary.
        milp_requirement = (
            ceiling_whole_units(
                source_values
            )
        )

        result[
            requirement_column
        ] = (
            milp_requirement
        )

        result[
            increment_column
        ] = (
            milp_requirement
            -
            source_values
        )

        result[
            net_column
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Int64",
        )

        eligible_requirement = (
            result.loc[
                eligible,
                requirement_column,
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        eligible_stock_integer = (
            result.loc[
                eligible,
                "MILP_Stock_Qty",
            ]
            .astype(
                np.int64
            )
            .to_numpy()
        )

        net_requirement = np.maximum(
            0,
            (
                eligible_requirement
                -
                eligible_stock_integer
            ),
        ).astype(
            np.int64
        )

        result.loc[
            eligible,
            net_column,
        ] = (
            net_requirement
        )

        result[
            value_column
        ] = np.nan

        result.loc[
            eligible,
            value_column,
        ] = (
            result.loc[
                eligible,
                net_column,
            ]
            .astype(float)
            *
            result.loc[
                eligible,
                "Planning_Cost",
            ]
            .astype(float)
        )

        result[
            f"MILP_Positive_Net_{scenario}"
        ] = False

        result.loc[
            eligible,
            f"MILP_Positive_Net_{scenario}",
        ] = (
            result.loc[
                eligible,
                net_column,
            ]
            .astype(
                np.int64
            )
            .gt(0)
        )

    # --------------------------------------------------------
    # EXPLICIT POINT-ROUNDING GOVERNANCE
    # --------------------------------------------------------

    result[
        "MILP_Point_Source_Was_Whole_Unit"
    ] = is_integer_like(
        result[
            "Planning_Requirement_Point"
        ]
        .to_numpy(
            dtype=float
        )
    )

    result[
        "MILP_Point_Rounding_Applied"
    ] = (
        ~result[
            "MILP_Point_Source_Was_Whole_Unit"
        ]
    )

    result[
        "MILP_Point_Rounding_Policy"
    ] = (
        "CEILING_TO_WHOLE_UNIT_IF_NEEDED"
    )

    result[
        "MILP_Point_Statistical_Forecast_Modified"
    ] = False

    result[
        "MILP_Decision_Variable_Domain"
    ] = np.where(
        eligible,
        "NONNEGATIVE_INTEGER",
        "NOT_IN_OPTIMIZATION",
    )

    result[
        "MILP_Decision_Unit"
    ] = (
        "INVENTORY_REPORT_UNIT"
    )

    result[
        "Supplier_Pack_Size_Constraint_Available"
    ] = False

    result[
        "Clinical_Priority_Used"
    ] = False

    result[
        "MILP_Input_Version"
    ] = (
        MILP_INPUT_VERSION
    )

    result[
        "MILP_Publication_Status"
    ] = (
        "CANDIDATE_NOT_PUBLISHED"
    )

    result[
        "MILP_Run_ID"
    ] = (
        RUNTIME_CONTEXT.run_id
    )

    result[
        "MILP_Run_Mode"
    ] = (
        RUNTIME_CONTEXT.run_mode
    )

    return result


def validate_milp_inputs(
    dataframe: pd.DataFrame,
    expected_rows: int,
) -> None:
    if len(
        dataframe
    ) != expected_rows:
        raise ValueError(
            "MILP-input population "
            "does not match source population."
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate Medicine_Key."
        )

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    if (
        dataframe.loc[
            eligible,
            "MILP_Stock_Qty",
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            "MILP stock missing "
            "on eligible row."
        )

    for scenario in SCENARIOS:
        requirement_column = (
            f"MILP_Requirement_{scenario}"
        )

        net_column = (
            f"MILP_Net_Requirement_{scenario}"
        )

        value_column = (
            f"MILP_Full_Funding_Value_{scenario}"
        )

        requirement = (
            dataframe[
                requirement_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not (
            is_integer_like(
                requirement
            )
            .all()
        ):
            raise ValueError(
                f"{requirement_column} "
                "is not whole-unit."
            )

        if (
            requirement
            < 0
        ).any():
            raise ValueError(
                f"Negative {requirement_column}."
            )

        if (
            dataframe.loc[
                eligible,
                net_column,
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                f"Missing eligible {net_column}."
            )

        net_values = (
            dataframe.loc[
                eligible,
                net_column,
            ]
            .astype(float)
            .to_numpy()
        )

        if not (
            is_integer_like(
                net_values
            )
            .all()
        ):
            raise ValueError(
                f"{net_column} "
                "is not whole-unit."
            )

        if (
            net_values
            < 0
        ).any():
            raise ValueError(
                f"Negative {net_column}."
            )

        if (
            dataframe.loc[
                eligible,
                value_column,
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                f"Missing eligible {value_column}."
            )

    if (
        dataframe[
            "MILP_Requirement_P80_FLOORED_CEIL"
        ]
        .lt(
            dataframe[
                "MILP_Requirement_POINT"
            ]
        )
        .any()
    ):
        raise ValueError(
            "Operational P80 requirement "
            "is below operational point."
        )

    if (
        dataframe[
            "MILP_Requirement_P90_FLOORED_CEIL"
        ]
        .lt(
            dataframe[
                "MILP_Requirement_POINT"
            ]
        )
        .any()
    ):
        raise ValueError(
            "Operational P90 requirement "
            "is below operational point."
        )

    if (
        dataframe[
            "MILP_Point_Statistical_Forecast_Modified"
        ]
        .any()
    ):
        raise ValueError(
            "Statistical point forecast "
            "was unexpectedly modified."
        )


def build_governance(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    return pd.DataFrame(
        [
            {
                "Planning_Year":
                    TARGET_YEAR,
                "Source_Table":
                    SOURCE_TABLE,
                "Population_Rows":
                    len(dataframe),
                "Strict_ABC_Eligible":
                    int(
                        eligible.sum()
                    ),
                "Point_Fractional_Source_Rows":
                    int(
                        dataframe[
                            "MILP_Point_Rounding_Applied"
                        ].sum()
                    ),
                "Point_Whole_Unit_Policy":
                    (
                        "CEILING_TO_WHOLE_UNIT_"
                        "IF_NEEDED"
                    ),
                "P80_Whole_Unit_Policy":
                    "CEILING_TO_WHOLE_UNIT",
                "P90_Whole_Unit_Policy":
                    "CEILING_TO_WHOLE_UNIT",
                "Statistical_Point_Forecast_Modified":
                    False,
                "Fractional_Stock_Automatically_Rounded":
                    False,
                "Supplier_Pack_Size_Constraint_Available":
                    False,
                "Clinical_Priority_Used":
                    False,
                "Publication_Status":
                    "CANDIDATE_NOT_PUBLISHED",
                "Run_ID":
                    RUNTIME_CONTEXT.run_id,
                "Run_Mode":
                    RUNTIME_CONTEXT.run_mode,
                "MILP_Input_Version":
                    MILP_INPUT_VERSION,
            }
        ]
    )


def write_database(
    dataframe: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:
    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:
        connection.register(
            "milp_df",
            dataframe,
        )

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {MILP_TABLE} AS
            SELECT *
            FROM milp_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {GOVERNANCE_TABLE} AS
            SELECT *
            FROM governance_df
            """
        )

    finally:
        connection.close()


def write_files(
    dataframe: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:
    dataframe.to_parquet(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_milp_inputs.parquet"
        ),
        index=False,
        engine="pyarrow",
    )

    dataframe.to_csv(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_milp_inputs.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_milp_governance.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "planning_year":
            TARGET_YEAR,
        "source_table":
            SOURCE_TABLE,
        "milp_table":
            MILP_TABLE,
        "whole_unit_point_policy":
            "CEILING_TO_WHOLE_UNIT_IF_NEEDED",
        "whole_unit_p80_policy":
            "CEILING_TO_WHOLE_UNIT",
        "whole_unit_p90_policy":
            "CEILING_TO_WHOLE_UNIT",
        "statistical_forecast_modified":
            False,
        "automatic_publication":
            False,
        "run_id":
            RUNTIME_CONTEXT.run_id,
        "run_mode":
            RUNTIME_CONTEXT.run_mode,
    }

    (
        OUTPUT_DIR
        / "milp_input_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 108)
    print("CDH INVENTORY DSS")
    print(
        f"{TARGET_YEAR} GOVERNED "
        "WHOLE-UNIT MILP INPUTS"
    )
    print("=" * 108)
    print()

    source = load_inputs()

    validate_source(
        source
    )

    print(
        "Source validation: PASS"
    )
    print(
        "Planning year:",
        TARGET_YEAR,
    )
    print(
        "Population:",
        f"{len(source):,}",
    )

    milp = build_milp_inputs(
        source
    )

    validate_milp_inputs(
        milp,
        len(source),
    )

    print(
        "MILP structural validation: PASS"
    )

    governance = build_governance(
        milp
    )

    write_files(
        milp,
        governance,
    )

    write_database(
        milp,
        governance,
    )

    eligible = int(
        milp[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    print()
    print(
        "Strict A/B/C eligible:",
        f"{eligible:,}",
    )

    print(
        "Fractional POINT rows ceiled "
        "for operational MILP:",
        int(
            milp[
                "MILP_Point_Rounding_Applied"
            ].sum()
        ),
    )

    print(
        "Statistical point forecast modified:",
        False,
    )

    print(
        "Automatically published:",
        False,
    )

    print()
    print("=" * 108)
    print(
        f"{TARGET_YEAR} GOVERNED "
        "WHOLE-UNIT MILP INPUTS COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
