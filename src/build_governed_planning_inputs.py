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

SOURCE_YEAR = int(
    RUNTIME_CONTEXT.latest_completed_year
)

HISTORICAL_START_YEAR = int(
    RUNTIME_CONTEXT.historical_start_year
)

PLANNING_VERSION = (
    f"PLANNING_{TARGET_YEAR}_"
    "GOVERNED_INPUTS_V1"
)

OPTIMIZER_INPUT_VERSION = (
    f"PLANNING_{TARGET_YEAR}_"
    "GOVERNED_OPTIMIZER_INPUTS_V1"
)

FORECAST_TABLE = (
    f"forecast_{TARGET_YEAR}_"
    "governed_candidate"
)

PLANNING_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_inputs"
)

OPTIMIZER_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_optimizer_inputs"
)

REVIEW_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_review_queue"
)

GOVERNANCE_TABLE = (
    f"planning_{TARGET_YEAR}_"
    "governed_governance"
)

OUTPUT_DIR = (
    RUNTIME_CONTEXT.processed_dir
    / "planning"
    / (
        f"planning_{TARGET_YEAR}_"
        "governed_v1"
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PRIORITY_CLASSES = [
    "A",
    "B",
    "C",
]

ABC_PRIORITY_ORDER = {
    "A": 1,
    "B": 2,
    "C": 3,
    "UNCLASSIFIED_COST": 99,
}


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        forecast = connection.execute(
            f"""
            SELECT *
            FROM {FORECAST_TABLE}
            ORDER BY Medicine_Key
            """
        ).df()

        fact = connection.execute(
            """
            SELECT
                Medicine_Key,
                Year,
                Brand_Name,
                Generic_Name,
                Unit,
                Ending_Balance,
                Average_cost,
                Cost_Valid,
                Data_Quality_Status,
                Source_File
            FROM fact_inventory_annual
            WHERE Year BETWEEN ? AND ?
            ORDER BY Medicine_Key, Year
            """,
            [
                HISTORICAL_START_YEAR,
                SOURCE_YEAR,
            ],
        ).df()

    finally:
        connection.close()

    return forecast, fact


def validate_inputs(
    forecast: pd.DataFrame,
    fact: pd.DataFrame,
) -> None:
    if forecast.empty:
        raise ValueError(
            "Governed candidate forecast is empty."
        )

    if (
        forecast[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate Medicine_Key "
            "in governed forecast."
        )

    if not (
        forecast[
            "Target_Year"
        ]
        .eq(
            TARGET_YEAR
        )
        .all()
    ):
        raise ValueError(
            "Forecast target year "
            "does not match runtime context."
        )

    if not (
        forecast[
            "Feature_Cutoff_Year"
        ]
        .eq(
            SOURCE_YEAR
        )
        .all()
    ):
        raise ValueError(
            "Forecast feature cutoff "
            "does not match latest "
            "completed year."
        )

    if not (
        forecast[
            "Publication_Status"
        ]
        .eq(
            "CANDIDATE_NOT_PUBLISHED"
        )
        .all()
    ):
        raise ValueError(
            "Planning must be built from "
            "an unpublished governed candidate."
        )

    duplicate_fact = int(
        fact.duplicated(
            subset=[
                "Medicine_Key",
                "Year",
            ]
        ).sum()
    )

    if duplicate_fact:
        raise ValueError(
            "Duplicate Medicine_Key-Year "
            "in historical fact."
        )


def build_stock(
    forecast: pd.DataFrame,
    fact: pd.DataFrame,
) -> pd.DataFrame:
    stock = (
        fact.loc[
            fact[
                "Year"
            ].eq(
                SOURCE_YEAR
            ),
            [
                "Medicine_Key",
                "Ending_Balance",
                "Data_Quality_Status",
                "Source_File",
            ],
        ]
        .copy()
    )

    stock = (
        forecast[
            [
                "Medicine_Key",
            ]
        ]
        .merge(
            stock,
            on="Medicine_Key",
            how="left",
            validate="one_to_one",
        )
    )

    stock = stock.rename(
        columns={
            "Ending_Balance":
                "Source_Ending_Balance",
            "Data_Quality_Status":
                "Stock_Source_Data_Quality_Status",
            "Source_File":
                "Stock_Source_File",
        }
    )

    missing = (
        stock[
            "Source_Ending_Balance"
        ]
        .isna()
    )

    negative = (
        stock[
            "Source_Ending_Balance"
        ]
        .lt(0)
        .fillna(False)
    )

    valid = (
        ~missing
        &
        ~negative
    )

    stock[
        "Planning_Stock_Status"
    ] = np.select(
        [
            missing,
            negative,
            valid,
        ],
        [
            "MISSING_COMPLETED_YEAR_ENDING_BALANCE",
            "NEGATIVE_COMPLETED_YEAR_ENDING_BALANCE",
            "VALID_COMPLETED_YEAR_ENDING_BALANCE",
        ],
        default="UNKNOWN",
    )

    stock[
        "Planning_Stock_Valid"
    ] = valid

    stock[
        "Planning_Stock_Qty"
    ] = np.nan

    stock.loc[
        valid,
        "Planning_Stock_Qty",
    ] = stock.loc[
        valid,
        "Source_Ending_Balance",
    ]

    stock[
        "Planning_Stock_Source_Year"
    ] = SOURCE_YEAR

    stock[
        "Planning_Stock_Source_Type"
    ] = (
        "COMPLETED_ANNUAL_ENDING_BALANCE"
    )

    return stock


def build_cost(
    forecast: pd.DataFrame,
    fact: pd.DataFrame,
) -> pd.DataFrame:
    history = (
        fact.loc[
            fact[
                "Year"
            ].le(
                SOURCE_YEAR
            )
        ]
        .copy()
    )

    valid_cost = (
        history[
            "Cost_Valid"
        ]
        .fillna(False)
        .astype(bool)
        &
        history[
            "Average_cost"
        ]
        .notna()
        &
        history[
            "Average_cost"
        ]
        .gt(0)
    )

    usable = (
        history.loc[
            valid_cost,
            [
                "Medicine_Key",
                "Year",
                "Average_cost",
                "Source_File",
            ],
        ]
        .sort_values(
            [
                "Medicine_Key",
                "Year",
            ]
        )
        .drop_duplicates(
            subset=[
                "Medicine_Key",
            ],
            keep="last",
        )
        .copy()
    )

    usable = usable.rename(
        columns={
            "Year":
                "Planning_Cost_Source_Year",
            "Average_cost":
                "Planning_Cost",
            "Source_File":
                "Planning_Cost_Source_File",
        }
    )

    cost = (
        forecast[
            [
                "Medicine_Key",
            ]
        ]
        .merge(
            usable,
            on="Medicine_Key",
            how="left",
            validate="one_to_one",
        )
    )

    cost[
        "Planning_Cost_Available"
    ] = (
        cost[
            "Planning_Cost"
        ]
        .notna()
        &
        cost[
            "Planning_Cost"
        ]
        .gt(0)
    )

    cost[
        "Planning_Cost_Fallback_Used"
    ] = (
        cost[
            "Planning_Cost_Available"
        ]
        &
        cost[
            "Planning_Cost_Source_Year"
        ]
        .lt(
            SOURCE_YEAR
        )
    )

    cost[
        "Planning_Cost_Status"
    ] = np.select(
        [
            ~cost[
                "Planning_Cost_Available"
            ],
            cost[
                "Planning_Cost_Source_Year"
            ]
            .eq(
                SOURCE_YEAR
            ),
            cost[
                "Planning_Cost_Source_Year"
            ]
            .lt(
                SOURCE_YEAR
            ),
        ],
        [
            "UNAVAILABLE",
            "VALID_LATEST_COMPLETED_YEAR_COST",
            "HISTORICAL_VALID_COST_FALLBACK",
        ],
        default="UNKNOWN",
    )

    cost[
        "Planning_Cost_Age_Years"
    ] = np.nan

    available = (
        cost[
            "Planning_Cost_Available"
        ]
    )

    cost.loc[
        available,
        "Planning_Cost_Age_Years",
    ] = (
        SOURCE_YEAR
        -
        cost.loc[
            available,
            "Planning_Cost_Source_Year",
        ]
    )

    cost[
        "Planning_Cost_Source_Type"
    ] = np.where(
        cost[
            "Planning_Cost_Available"
        ],
        "LATEST_VALID_ANNUAL_AVERAGE_COST",
        "NONE",
    )

    return cost


def build_planning(
    forecast: pd.DataFrame,
    stock: pd.DataFrame,
    cost: pd.DataFrame,
) -> pd.DataFrame:
    result = forecast.merge(
        stock,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        cost,
        on="Medicine_Key",
        how="left",
        validate="one_to_one",
    )

    result[
        "Requirement_Point"
    ] = result[
        "Point_Forecast"
    ]

    result[
        "Requirement_P80"
    ] = result[
        "P80"
    ]

    result[
        "Requirement_P90"
    ] = result[
        "P90"
    ]

    result[
        "Flag_P80_Below_Point"
    ] = (
        result[
            "Requirement_P80"
        ]
        <
        result[
            "Requirement_Point"
        ]
    )

    result[
        "Flag_P90_Below_Point"
    ] = (
        result[
            "Requirement_P90"
        ]
        <
        result[
            "Requirement_Point"
        ]
    )

    stock_valid = (
        result[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    cost_valid = (
        result[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    for scenario in [
        "Point",
        "P80",
        "P90",
    ]:
        requirement_column = (
            f"Requirement_{scenario}"
        )

        net_column = (
            f"Net_Requirement_{scenario}"
        )

        value_column = (
            f"Estimated_Procurement_Value_{scenario}"
        )

        result[
            net_column
        ] = np.nan

        result.loc[
            stock_valid,
            net_column,
        ] = np.maximum(
            0,
            (
                result.loc[
                    stock_valid,
                    requirement_column,
                ]
                -
                result.loc[
                    stock_valid,
                    "Planning_Stock_Qty",
                ]
            ),
        )

        value_ready = (
            stock_valid
            &
            cost_valid
        )

        result[
            value_column
        ] = np.nan

        result.loc[
            value_ready,
            value_column,
        ] = (
            result.loc[
                value_ready,
                net_column,
            ]
            *
            result.loc[
                value_ready,
                "Planning_Cost",
            ]
        )

    result[
        "Optimizer_Core_Input_Ready"
    ] = (
        stock_valid
        &
        cost_valid
    )

    stock_problem = (
        ~stock_valid
    )

    cost_problem = (
        ~cost_valid
    )

    result[
        "Planning_Input_Status"
    ] = np.select(
        [
            (
                ~stock_problem
                &
                ~cost_problem
            ),
            (
                stock_problem
                &
                ~cost_problem
            ),
            (
                ~stock_problem
                &
                cost_problem
            ),
            (
                stock_problem
                &
                cost_problem
            ),
        ],
        [
            "READY",
            "STOCK_REVIEW_REQUIRED",
            "COST_REVIEW_REQUIRED",
            "STOCK_AND_COST_REVIEW_REQUIRED",
        ],
        default="UNKNOWN",
    )

    result[
        "Needs_Current_Stock_Override"
    ] = stock_problem

    result[
        "Needs_Planning_Cost_Override"
    ] = cost_problem

    result[
        "Planning_Stock_Is_Current_Midyear"
    ] = False

    result[
        "Year_End_Pharmacy_Balance_Used"
    ] = False

    result[
        "Year_End_Implied_Cost_Used"
    ] = False

    result[
        "Last_Purchase_Price_Used"
    ] = False

    result[
        "Historical_ABC_Cost_Modified"
    ] = False

    result[
        "Planning_Version"
    ] = PLANNING_VERSION

    result[
        "Planning_Run_ID"
    ] = (
        RUNTIME_CONTEXT.run_id
    )

    result[
        "Planning_Run_Mode"
    ] = (
        RUNTIME_CONTEXT.run_mode
    )

    result[
        "Planning_Publication_Status"
    ] = (
        "CANDIDATE_NOT_PUBLISHED"
    )

    return result


def validate_planning(
    dataframe: pd.DataFrame,
    expected_rows: int,
) -> None:
    if len(
        dataframe
    ) != expected_rows:
        raise ValueError(
            "Planning population differs "
            "from forecast population."
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate Medicine_Key "
            "in planning inputs."
        )

    valid_stock = (
        dataframe[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    valid_cost = (
        dataframe[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    if (
        dataframe.loc[
            valid_stock,
            "Planning_Stock_Qty",
        ]
        .lt(0)
        .any()
    ):
        raise ValueError(
            "Negative valid planning stock."
        )

    if (
        dataframe.loc[
            valid_cost,
            "Planning_Cost",
        ]
        .le(0)
        .any()
    ):
        raise ValueError(
            "Non-positive available "
            "planning cost."
        )

    for scenario in [
        "Point",
        "P80",
        "P90",
    ]:
        requirement = (
            f"Requirement_{scenario}"
        )

        net = (
            f"Net_Requirement_{scenario}"
        )

        value = (
            f"Estimated_Procurement_Value_{scenario}"
        )

        if (
            dataframe[
                requirement
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                f"Missing {requirement}."
            )

        if (
            dataframe[
                requirement
            ]
            .lt(0)
            .any()
        ):
            raise ValueError(
                f"Negative {requirement}."
            )

        if (
            dataframe.loc[
                valid_stock,
                net,
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                f"Missing valid-stock {net}."
            )

        if (
            dataframe.loc[
                valid_stock,
                net,
            ]
            .lt(0)
            .any()
        ):
            raise ValueError(
                f"Negative {net}."
            )

        ready = (
            valid_stock
            &
            valid_cost
        )

        if (
            dataframe.loc[
                ready,
                value,
            ]
            .isna()
            .any()
        ):
            raise ValueError(
                f"Missing ready {value}."
            )

        if (
            dataframe.loc[
                ready,
                value,
            ]
            .lt(0)
            .any()
        ):
            raise ValueError(
                f"Negative {value}."
            )

    if (
        dataframe[
            "Year_End_Pharmacy_Balance_Used"
        ]
        .any()
    ):
        raise ValueError(
            "Year-end pharmacy balance "
            "was unexpectedly used."
        )

    if (
        dataframe[
            "Year_End_Implied_Cost_Used"
        ]
        .any()
    ):
        raise ValueError(
            "Year-end implied cost "
            "was unexpectedly used."
        )

    if (
        dataframe[
            "Last_Purchase_Price_Used"
        ]
        .any()
    ):
        raise ValueError(
            "Last Purchase Price "
            "was unexpectedly used."
        )


def build_optimizer_inputs(
    planning: pd.DataFrame,
) -> pd.DataFrame:
    result = planning.copy()

    result[
        "Raw_Requirement_Point"
    ] = result[
        "Requirement_Point"
    ]

    result[
        "Raw_Requirement_P80"
    ] = result[
        "Requirement_P80"
    ]

    result[
        "Raw_Requirement_P90"
    ] = result[
        "Requirement_P90"
    ]

    result[
        "Planning_Requirement_Point"
    ] = result[
        "Raw_Requirement_Point"
    ]

    result[
        "Planning_Requirement_P80_Floored"
    ] = np.maximum(
        result[
            "Raw_Requirement_Point"
        ],
        result[
            "Raw_Requirement_P80"
        ],
    )

    result[
        "Planning_Requirement_P90_Floored"
    ] = np.maximum(
        result[
            "Raw_Requirement_Point"
        ],
        result[
            "Raw_Requirement_P90"
        ],
    )

    result[
        "P80_Incumbent_Floor_Applied"
    ] = (
        result[
            "Raw_Requirement_P80"
        ]
        <
        result[
            "Raw_Requirement_Point"
        ]
    )

    result[
        "P90_Incumbent_Floor_Applied"
    ] = (
        result[
            "Raw_Requirement_P90"
        ]
        <
        result[
            "Raw_Requirement_Point"
        ]
    )

    core_ready = (
        result[
            "Optimizer_Core_Input_Ready"
        ]
        .fillna(False)
        .astype(bool)
    )

    abc_classifiable = (
        result[
            "ABC_AsOf"
        ]
        .isin(
            PRIORITY_CLASSES
        )
    )

    result[
        "Optimization_Priority_Eligible"
    ] = (
        core_ready
        &
        abc_classifiable
    )

    result[
        "Ready_But_ABC_Unclassified"
    ] = (
        core_ready
        &
        result[
            "ABC_AsOf"
        ]
        .eq(
            "UNCLASSIFIED_COST"
        )
    )

    result[
        "High_Priority_AB_Unresolved"
    ] = (
        ~core_ready
        &
        result[
            "ABC_AsOf"
        ]
        .isin(
            [
                "A",
                "B",
            ]
        )
    )

    stock_valid = (
        result[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    cost_valid = (
        result[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    result[
        "Optimization_Eligibility_Status"
    ] = np.select(
        [
            (
                core_ready
                &
                abc_classifiable
            ),
            (
                core_ready
                &
                result[
                    "ABC_AsOf"
                ]
                .eq(
                    "UNCLASSIFIED_COST"
                )
            ),
            (
                ~stock_valid
                &
                cost_valid
            ),
            (
                stock_valid
                &
                ~cost_valid
            ),
            (
                ~stock_valid
                &
                ~cost_valid
            ),
        ],
        [
            "ELIGIBLE_ABC_PRIORITY",
            "READY_BUT_ABC_UNCLASSIFIED",
            "EXCLUDED_STOCK_REVIEW",
            "EXCLUDED_COST_REVIEW",
            "EXCLUDED_STOCK_AND_COST_REVIEW",
        ],
        default="UNKNOWN",
    )

    result[
        "ABC_Priority_Order"
    ] = (
        result[
            "ABC_AsOf"
        ]
        .map(
            ABC_PRIORITY_ORDER
        )
        .fillna(99)
        .astype(int)
    )

    requirement_map = {
        "POINT":
            "Planning_Requirement_Point",
        "P80_FLOORED":
            "Planning_Requirement_P80_Floored",
        "P90_FLOORED":
            "Planning_Requirement_P90_Floored",
    }

    for scenario, requirement_column in (
        requirement_map.items()
    ):
        net_column = (
            f"Optimization_Net_"
            f"Requirement_{scenario}"
        )

        value_column = (
            f"Optimization_Full_"
            f"Funding_Value_{scenario}"
        )

        result[
            net_column
        ] = np.nan

        result.loc[
            stock_valid,
            net_column,
        ] = np.maximum(
            0,
            (
                result.loc[
                    stock_valid,
                    requirement_column,
                ]
                -
                result.loc[
                    stock_valid,
                    "Planning_Stock_Qty",
                ]
            ),
        )

        result[
            value_column
        ] = np.nan

        eligible = (
            result[
                "Optimization_Priority_Eligible"
            ]
        )

        result.loc[
            eligible,
            value_column,
        ] = (
            result.loc[
                eligible,
                net_column,
            ]
            *
            result.loc[
                eligible,
                "Planning_Cost",
            ]
        )

    result[
        "Point_Requirement_Policy"
    ] = (
        "GOVERNED_POINT_FORECAST"
    )

    result[
        "P80_Requirement_Policy"
    ] = (
        "MAX_GOVERNED_POINT_AND_RAW_P80"
    )

    result[
        "P90_Requirement_Policy"
    ] = (
        "MAX_GOVERNED_POINT_AND_RAW_P90"
    )

    result[
        "Raw_P80_Remains_Statistical_Quantile"
    ] = True

    result[
        "Raw_P90_Remains_Statistical_Quantile"
    ] = True

    result[
        "Floored_P80_Is_Statistical_Quantile"
    ] = False

    result[
        "Floored_P90_Is_Statistical_Quantile"
    ] = False

    result[
        "Optimizer_Input_Version"
    ] = (
        OPTIMIZER_INPUT_VERSION
    )

    result[
        "Optimizer_Publication_Status"
    ] = (
        "CANDIDATE_NOT_PUBLISHED"
    )

    return result


def validate_optimizer(
    dataframe: pd.DataFrame,
    expected_rows: int,
) -> None:
    if len(
        dataframe
    ) != expected_rows:
        raise ValueError(
            "Optimizer-input population "
            "does not match planning population."
        )

    if (
        dataframe[
            "Medicine_Key"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate Medicine_Key "
            "in optimizer inputs."
        )

    if (
        dataframe[
            "Planning_Requirement_P80_Floored"
        ]
        .lt(
            dataframe[
                "Planning_Requirement_Point"
            ]
        )
        .any()
    ):
        raise ValueError(
            "Floored P80 below point forecast."
        )

    if (
        dataframe[
            "Planning_Requirement_P90_Floored"
        ]
        .lt(
            dataframe[
                "Planning_Requirement_Point"
            ]
        )
        .any()
    ):
        raise ValueError(
            "Floored P90 below point forecast."
        )

    eligible = (
        dataframe[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    if (
        ~dataframe.loc[
            eligible,
            "ABC_AsOf",
        ]
        .isin(
            PRIORITY_CLASSES
        )
    ).any():
        raise ValueError(
            "Invalid ABC class entered "
            "priority-eligible population."
        )

    if (
        ~dataframe.loc[
            eligible,
            "Planning_Stock_Valid",
        ]
        .fillna(False)
        .astype(bool)
    ).any():
        raise ValueError(
            "Eligible row lacks valid stock."
        )

    if (
        ~dataframe.loc[
            eligible,
            "Planning_Cost_Available",
        ]
        .fillna(False)
        .astype(bool)
    ).any():
        raise ValueError(
            "Eligible row lacks planning cost."
        )

    for scenario in [
        "POINT",
        "P80_FLOORED",
        "P90_FLOORED",
    ]:
        net_column = (
            f"Optimization_Net_"
            f"Requirement_{scenario}"
        )

        value_column = (
            f"Optimization_Full_"
            f"Funding_Value_{scenario}"
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

        if (
            dataframe.loc[
                eligible,
                net_column,
            ]
            .lt(0)
            .any()
        ):
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


def build_review_queue(
    optimizer: pd.DataFrame,
) -> pd.DataFrame:
    review = (
        optimizer.loc[
            ~optimizer[
                "Optimizer_Core_Input_Ready"
            ]
            .fillna(False)
            .astype(bool)
        ]
        .copy()
    )

    review[
        "Needs_Stock_Review"
    ] = (
        ~review[
            "Planning_Stock_Valid"
        ]
        .fillna(False)
        .astype(bool)
    )

    review[
        "Needs_Cost_Review"
    ] = (
        ~review[
            "Planning_Cost_Available"
        ]
        .fillna(False)
        .astype(bool)
    )

    review[
        "Maximum_Planning_Requirement"
    ] = (
        review[
            [
                "Requirement_Point",
                "Requirement_P80",
                "Requirement_P90",
            ]
        ]
        .max(
            axis=1
        )
    )

    review[
        "High_Priority_ABC_Review"
    ] = (
        review[
            "ABC_AsOf"
        ]
        .isin(
            [
                "A",
                "B",
            ]
        )
    )

    review[
        "Full_Optimization_Blocker"
    ] = (
        review[
            "High_Priority_ABC_Review"
        ]
    )

    review = (
        review.sort_values(
            [
                "ABC_Priority_Order",
                "Full_Optimization_Blocker",
                "Maximum_Planning_Requirement",
                "Medicine_Key",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    review[
        "Review_Queue_Order"
    ] = np.arange(
        1,
        len(review) + 1,
    )

    return review


def build_governance(
    planning: pd.DataFrame,
    optimizer: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    core_ready = int(
        planning[
            "Optimizer_Core_Input_Ready"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    eligible = int(
        optimizer[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    ready_unclassified = int(
        optimizer[
            "Ready_But_ABC_Unclassified"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    stock_review = int(
        (
            ~planning[
                "Planning_Stock_Valid"
            ]
            .fillna(False)
            .astype(bool)
        ).sum()
    )

    cost_review = int(
        (
            ~planning[
                "Planning_Cost_Available"
            ]
            .fillna(False)
            .astype(bool)
        ).sum()
    )

    return pd.DataFrame(
        [
            {
                "Planning_Year":
                    TARGET_YEAR,
                "Source_Completed_Year":
                    SOURCE_YEAR,
                "Historical_Start_Year":
                    HISTORICAL_START_YEAR,
                "Forecast_Table":
                    FORECAST_TABLE,
                "Forecast_Model":
                    (
                        planning[
                            "Point_Forecast_Model"
                        ]
                        .iloc[0]
                    ),
                "Planning_Rows":
                    len(planning),
                "Optimizer_Core_Ready":
                    core_ready,
                "Strict_ABC_Eligible":
                    eligible,
                "Ready_But_ABC_Unclassified":
                    ready_unclassified,
                "Stock_Review_Rows":
                    stock_review,
                "Cost_Review_Rows":
                    cost_review,
                "Review_Queue_Rows":
                    len(review),
                "Point_Requirement_Policy":
                    "GOVERNED_POINT_FORECAST",
                "P80_Requirement_Policy":
                    "MAX_GOVERNED_POINT_AND_RAW_P80",
                "P90_Requirement_Policy":
                    "MAX_GOVERNED_POINT_AND_RAW_P90",
                "Planning_Stock_Policy":
                    (
                        "LATEST_COMPLETED_YEAR_"
                        "VALID_NONNEGATIVE_ENDING_BALANCE"
                    ),
                "Planning_Cost_Policy":
                    (
                        "LATEST_VALID_ANNUAL_"
                        "AVERAGE_COST_BACKWARD"
                    ),
                "Missing_Stock_Assumed_Zero":
                    False,
                "Missing_Cost_Assumed_Zero":
                    False,
                "Clinical_Priority_Used":
                    False,
                "Publication_Status":
                    "CANDIDATE_NOT_PUBLISHED",
                "Run_ID":
                    RUNTIME_CONTEXT.run_id,
                "Run_Mode":
                    RUNTIME_CONTEXT.run_mode,
                "Planning_Version":
                    PLANNING_VERSION,
                "Optimizer_Input_Version":
                    OPTIMIZER_INPUT_VERSION,
            }
        ]
    )


def write_database(
    planning: pd.DataFrame,
    optimizer: pd.DataFrame,
    review: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:
    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:
        connection.register(
            "planning_df",
            planning,
        )

        connection.register(
            "optimizer_df",
            optimizer,
        )

        connection.register(
            "review_df",
            review,
        )

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {PLANNING_TABLE} AS
            SELECT *
            FROM planning_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {OPTIMIZER_TABLE} AS
            SELECT *
            FROM optimizer_df
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE
            {REVIEW_TABLE} AS
            SELECT *
            FROM review_df
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
    planning: pd.DataFrame,
    optimizer: pd.DataFrame,
    review: pd.DataFrame,
    governance: pd.DataFrame,
) -> None:
    planning.to_parquet(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_inputs.parquet"
        ),
        index=False,
        engine="pyarrow",
    )

    planning.to_csv(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_inputs.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    optimizer.to_parquet(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_optimizer_inputs.parquet"
        ),
        index=False,
        engine="pyarrow",
    )

    optimizer.to_csv(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_optimizer_inputs.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    review.to_csv(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governed_review_queue.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    governance.to_csv(
        OUTPUT_DIR
        / (
            f"planning_{TARGET_YEAR}_"
            "governance.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "planning_year":
            TARGET_YEAR,
        "source_completed_year":
            SOURCE_YEAR,
        "historical_start_year":
            HISTORICAL_START_YEAR,
        "forecast_table":
            FORECAST_TABLE,
        "planning_table":
            PLANNING_TABLE,
        "optimizer_table":
            OPTIMIZER_TABLE,
        "review_table":
            REVIEW_TABLE,
        "publication_status":
            "CANDIDATE_NOT_PUBLISHED",
        "automatic_publication":
            False,
        "run_id":
            RUNTIME_CONTEXT.run_id,
        "run_mode":
            RUNTIME_CONTEXT.run_mode,
    }

    (
        OUTPUT_DIR
        / "planning_metadata.json"
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
        "PLANNING AND READINESS"
    )
    print("=" * 108)
    print()

    forecast, fact = load_data()

    validate_inputs(
        forecast,
        fact,
    )

    print(
        "Input validation: PASS"
    )
    print(
        "Planning year:",
        TARGET_YEAR,
    )
    print(
        "Completed-year stock/cost source:",
        SOURCE_YEAR,
    )
    print(
        "Forecast population:",
        f"{len(forecast):,}",
    )

    stock = build_stock(
        forecast,
        fact,
    )

    cost = build_cost(
        forecast,
        fact,
    )

    planning = build_planning(
        forecast,
        stock,
        cost,
    )

    validate_planning(
        planning,
        len(forecast),
    )

    print(
        "Planning input validation: PASS"
    )

    optimizer = (
        build_optimizer_inputs(
            planning
        )
    )

    validate_optimizer(
        optimizer,
        len(planning),
    )

    print(
        "Optimizer readiness validation: PASS"
    )

    review = (
        build_review_queue(
            optimizer
        )
    )

    governance = (
        build_governance(
            planning,
            optimizer,
            review,
        )
    )

    write_files(
        planning,
        optimizer,
        review,
        governance,
    )

    write_database(
        planning,
        optimizer,
        review,
        governance,
    )

    core_ready = int(
        planning[
            "Optimizer_Core_Input_Ready"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    strict_eligible = int(
        optimizer[
            "Optimization_Priority_Eligible"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    ready_unclassified = int(
        optimizer[
            "Ready_But_ABC_Unclassified"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    print()
    print(
        "Optimizer core ready:",
        f"{core_ready:,}",
    )
    print(
        "Strict A/B/C eligible:",
        f"{strict_eligible:,}",
    )
    print(
        "Ready but ABC unclassified:",
        f"{ready_unclassified:,}",
    )
    print(
        "Review queue:",
        f"{len(review):,}",
    )
    print(
        "P80 floor applications:",
        int(
            optimizer[
                "P80_Incumbent_Floor_Applied"
            ].sum()
        ),
    )
    print(
        "P90 floor applications:",
        int(
            optimizer[
                "P90_Incumbent_Floor_Applied"
            ].sum()
        ),
    )
    print(
        "Automatically published:",
        False,
    )

    print()
    print("=" * 108)
    print(
        f"{TARGET_YEAR} GOVERNED "
        "PLANNING AND READINESS COMPLETE"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
