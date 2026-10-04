from __future__ import annotations

from pathlib import Path

import json
import os

import duckdb
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_DIR = (
    PROJECT_ROOT
    / "database"
)

DEFAULT_DATABASE_PATH = (
    DATABASE_DIR
    / "cdh_inventory.duckdb"
)

VERSIONS_DIR = (
    DATABASE_DIR
    / "versions"
)

CURRENT_POINTER_PATH = (
    DATABASE_DIR
    / "current.json"
)


def _resolve_environment_database_path(
    configured: str,
) -> Path:

    path = Path(
        configured
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def _resolve_pointer_database_path() -> Path | None:

    if not CURRENT_POINTER_PATH.exists():
        return None

    try:
        payload = json.loads(
            CURRENT_POINTER_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise RuntimeError(
            "The CURRENT database pointer "
            "could not be read."
        ) from error

    if (
        payload.get("Pointer_Version")
        != "CURRENT_DATABASE_POINTER_V1"
    ):
        raise RuntimeError(
            "Unsupported CURRENT database "
            "pointer version."
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
            "CURRENT database pointer is "
            "missing Database_Relative_Path."
        )

    relative_path = Path(
        relative_value
    )

    if relative_path.is_absolute():
        raise RuntimeError(
            "CURRENT database pointer must "
            "use a relative version path."
        )

    candidate = (
        DATABASE_DIR
        / relative_path
    ).resolve()

    versions_root = (
        VERSIONS_DIR.resolve()
    )

    try:
        candidate.relative_to(
            versions_root
        )
    except ValueError as error:
        raise RuntimeError(
            "CURRENT database pointer must "
            "resolve inside database/versions."
        ) from error

    if candidate.suffix.lower() != ".duckdb":
        raise RuntimeError(
            "CURRENT database pointer must "
            "reference a DuckDB snapshot."
        )

    if not candidate.exists():
        raise FileNotFoundError(
            "Published CURRENT DuckDB snapshot "
            f"not found: {candidate}"
        )

    return candidate


def resolve_database_path() -> Path:

    configured = os.getenv(
        "CDH_APP_DATABASE_PATH"
    )

    # Explicit test/development override has
    # highest precedence.
    if configured:
        return (
            _resolve_environment_database_path(
                configured
            )
        )

    # Governed deployment uses an atomic pointer
    # to an immutable versioned snapshot.
    pointed = (
        _resolve_pointer_database_path()
    )

    if pointed is not None:
        return pointed

    # Backward-compatible fallback before the
    # first governed publication is performed.
    return DEFAULT_DATABASE_PATH


def get_database_path() -> Path:

    return resolve_database_path()


# Compatibility constant. New application reads
# resolve the active path at connection time.
DATABASE_PATH = resolve_database_path()


# ============================================================
# DATABASE HELPERS
# ============================================================

def connect_read_only() -> duckdb.DuckDBPyConnection:

    database_path = (
        resolve_database_path()
    )

    if not database_path.exists():
        raise FileNotFoundError(
            "DuckDB database not found: "
            f"{database_path}"
        )

    return duckdb.connect(
        str(database_path),
        read_only=True,
    )


def query_df(
    sql: str,
    params=None,
) -> pd.DataFrame:

    connection = connect_read_only()

    try:
        if params is None:
            return connection.execute(sql).df()

        return connection.execute(
            sql,
            params,
        ).df()

    finally:
        connection.close()


def scalar(
    sql: str,
    params=None,
):

    connection = connect_read_only()

    try:
        if params is None:
            result = connection.execute(
                sql
            ).fetchone()
        else:
            result = connection.execute(
                sql,
                params,
            ).fetchone()

    finally:
        connection.close()

    if result is None:
        return None

    return result[0]


def table_exists(
    table_name: str,
) -> bool:

    result = scalar(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?
        """,
        [table_name],
    )

    return bool(result)


# ============================================================
# CURRENT SYSTEM CONTEXT
# ============================================================

def get_system_context() -> dict:

    row = query_df(
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
    ).iloc[0]

    return {
        "Contract_Version":
            str(row["Contract_Version"]),

        "Historical_Start_Year":
            int(row["Historical_Start_Year"]),

        "Latest_Completed_Year":
            int(row["Latest_Completed_Year"]),

        "Planning_Year":
            int(row["Planning_Year"]),

        "Candidate_Run_ID":
            str(row["Candidate_Run_ID"]),

        "Build_Run_Mode":
            str(row["Build_Run_Mode"]),

        "Current_State":
            str(row["Current_State"]),

        "Published_Version":
            (
                None
                if pd.isna(
                    row["Published_Version"]
                )
                else str(
                    row["Published_Version"]
                )
            ),

        "Publication_Status":
            str(row["Publication_Status"]),

        "Automatic_Publication":
            bool(row["Automatic_Publication"]),
    }


# ============================================================
# CORE DSS TABLES
# ============================================================

CORE_TABLES = [
    "fact_inventory_annual",
    "abc_classification_annual",
    "current_forecast_segment_summary",
    "current_forecast",
    "current_planning_inputs",
    "current_milp_budget_scenarios",
    "retrospective_2025_matched_budget_summary",
    "expiry_dss_medicine_summary",
]


def get_core_table_health() -> pd.DataFrame:

    rows = []

    connection = connect_read_only()

    try:
        existing = {
            row[0]
            for row in connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }

        for table in CORE_TABLES:

            exists = table in existing

            if exists:
                count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            else:
                count = None

            rows.append(
                {
                    "Table": table,
                    "Available": exists,
                    "Rows": count,
                }
            )

    finally:
        connection.close()

    return pd.DataFrame(rows)


def get_executive_snapshot() -> dict:

    connection = connect_read_only()

    try:

        planning = connection.execute(
            """
            SELECT
                COUNT(*) AS Planning_Medicines,

                SUM(
                    CASE
                        WHEN Optimizer_Core_Input_Ready
                        THEN 1 ELSE 0
                    END
                ) AS Optimizer_Ready,

                SUM(
                    CASE
                        WHEN Needs_Current_Stock_Override
                        THEN 1 ELSE 0
                    END
                ) AS Needs_Stock_Override,

                SUM(
                    CASE
                        WHEN Needs_Planning_Cost_Override
                        THEN 1 ELSE 0
                    END
                ) AS Needs_Cost_Override,

                SUM(Point_Forecast)
                    AS Total_Point_Forecast,

                SUM(
                    CASE
                        WHEN ABC_AsOf = 'A'
                        THEN 1 ELSE 0
                    END
                ) AS ABC_A,

                SUM(
                    CASE
                        WHEN ABC_AsOf = 'B'
                        THEN 1 ELSE 0
                    END
                ) AS ABC_B,

                SUM(
                    CASE
                        WHEN ABC_AsOf = 'C'
                        THEN 1 ELSE 0
                    END
                ) AS ABC_C,

                SUM(
                    CASE
                        WHEN ABC_AsOf = 'UNCLASSIFIED_COST'
                        THEN 1 ELSE 0
                    END
                ) AS ABC_Unclassified_Cost

            FROM current_planning_inputs
            """
        ).fetchone()

        expiry = connection.execute(
            """
            SELECT
                SUM(
                    CASE
                        WHEN Has_Authoritative_Expiry_History
                        THEN 1 ELSE 0
                    END
                ) AS With_Expiry_History,

                SUM(
                    CASE
                        WHEN NOT Has_Authoritative_Expiry_History
                        THEN 1 ELSE 0
                    END
                ) AS Without_Expiry_History

            FROM expiry_dss_medicine_summary
            """
        ).fetchone()

        retrospective = connection.execute(
            """
            SELECT COUNT(*)
            FROM retrospective_2025_matched_budget_summary
            """
        ).fetchone()[0]

    finally:
        connection.close()

    return {
        "Planning_Medicines": int(planning[0]),
        "Optimizer_Ready": int(planning[1]),
        "Needs_Stock_Override": int(planning[2]),
        "Needs_Cost_Override": int(planning[3]),
        "Total_Point_Forecast": float(planning[4]),
        "ABC_A": int(planning[5]),
        "ABC_B": int(planning[6]),
        "ABC_C": int(planning[7]),
        "ABC_Unclassified_Cost": int(planning[8]),
        "With_Expiry_History": int(expiry[0]),
        "Without_Expiry_History": int(expiry[1]),
        "Retrospective_Scenario_Rows": int(
            retrospective
        ),
    }


# ============================================================
# INVENTORY & DEMAND PROFILE
# ============================================================

def get_annual_inventory_profile() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            Year,

            COUNT(*) AS Medicines,

            SUM(Annual_Issuance)
                AS Total_Issuance,

            SUM(
                CASE
                    WHEN Annual_Issuance > 0
                    THEN 1 ELSE 0
                END
            ) AS Active_Medicines,

            SUM(
                CASE
                    WHEN Annual_Issuance = 0
                    THEN 1 ELSE 0
                END
            ) AS Zero_Issuance_Medicines,

            SUM(
                CASE
                    WHEN ACV_Computable
                    THEN Annual_Consumption_Value
                    ELSE 0
                END
            ) AS Known_ACV

        FROM fact_inventory_annual

        GROUP BY Year

        ORDER BY Year
        """
    )



def get_latest_abc_profile() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            a.ABC_Class,
            a.ABC_Eligibility,

            COUNT(*) AS Medicine_Count,

            SUM(
                CASE
                    WHEN a.ACV_Computable
                    THEN a.Annual_Consumption_Value
                    ELSE 0
                END
            ) AS Known_ACV,

            SUM(
                CASE
                    WHEN a.ACV_Computable
                    THEN a.Annual_Consumption_Value
                    ELSE 0
                END
            )
            /
            NULLIF(
                SUM(
                    SUM(
                        CASE
                            WHEN a.ACV_Computable
                            THEN a.Annual_Consumption_Value
                            ELSE 0
                        END
                    )
                ) OVER (),
                0
            ) AS Known_ACV_Share

        FROM abc_classification_annual a

        WHERE a.Year = (
            SELECT Latest_Completed_Year
            FROM current_system_context
        )

        GROUP BY
            a.ABC_Class,
            a.ABC_Eligibility

        ORDER BY
            CASE a.ABC_Class
                WHEN 'A' THEN 1
                WHEN 'B' THEN 2
                WHEN 'C' THEN 3
                ELSE 4
            END
        """
    )



def get_demand_behavior_summary() -> pd.DataFrame:

    return query_df(
        """
        WITH context AS (
            SELECT
                Historical_Start_Year,
                Latest_Completed_Year
            FROM current_system_context
        ),

        historical AS (
            SELECT
                f.Medicine_Key,

                SUM(
                    COALESCE(
                        f.Annual_Issuance,
                        0
                    )
                ) AS Historical_Total_Issuance

            FROM fact_inventory_annual f
            CROSS JOIN context c

            WHERE f.Year BETWEEN
                c.Historical_Start_Year
                AND c.Latest_Completed_Year

            GROUP BY
                f.Medicine_Key
        ),

        base AS (
            SELECT
                p.Medicine_Key,

                p.Behavior_Segment_AsOf
                    AS Behavior_Segment,

                p.Persistence_AsOf
                    AS Persistence,

                b.CV_Positive_Issuance,

                COALESCE(
                    h.Historical_Total_Issuance,
                    0
                ) AS Historical_Total_Issuance

            FROM current_planning_inputs p

            LEFT JOIN historical h
                ON p.Medicine_Key =
                   h.Medicine_Key

            LEFT JOIN abc_behavior_annual b
                ON p.Medicine_Key =
                   b.Medicine_Key
               AND b.Year = (
                    SELECT Latest_Completed_Year
                    FROM context
               )
        )

        SELECT
            Behavior_Segment,

            COUNT(*)::BIGINT
                AS Medicine_Count,

            (
                100.0
                * COUNT(*)
                /
                NULLIF(
                    SUM(COUNT(*)) OVER (),
                    0
                )
            ) AS Medicine_Share_Pct,

            SUM(
                Historical_Total_Issuance
            ) AS Historical_Total_Issuance,

            (
                100.0
                * SUM(
                    Historical_Total_Issuance
                )
                /
                NULLIF(
                    SUM(
                        SUM(
                            Historical_Total_Issuance
                        )
                    ) OVER (),
                    0
                )
            ) AS Issuance_Share_Pct,

            MEDIAN(
                Persistence
            ) AS Median_Persistence,

            MEDIAN(
                CV_Positive_Issuance
            ) AS Median_Positive_CV

        FROM base

        GROUP BY
            Behavior_Segment

        ORDER BY
            CASE Behavior_Segment
                WHEN 'PERSISTENT_STABLE'
                    THEN 1
                WHEN 'PERSISTENT_VARIABLE'
                    THEN 2
                WHEN 'SPARSE'
                    THEN 3
                WHEN 'NON_MOVING'
                    THEN 4
                ELSE 5
            END
        """
    )



def get_latest_abc_behavior_matrix() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            ABC_AsOf
                AS ABC_Class,

            SUM(
                CASE
                    WHEN Behavior_Segment_AsOf =
                         'NON_MOVING'
                    THEN 1 ELSE 0
                END
            )::BIGINT
                AS NON_MOVING,

            SUM(
                CASE
                    WHEN Behavior_Segment_AsOf =
                         'PERSISTENT_STABLE'
                    THEN 1 ELSE 0
                END
            )::BIGINT
                AS PERSISTENT_STABLE,

            SUM(
                CASE
                    WHEN Behavior_Segment_AsOf =
                         'PERSISTENT_VARIABLE'
                    THEN 1 ELSE 0
                END
            )::BIGINT
                AS PERSISTENT_VARIABLE,

            SUM(
                CASE
                    WHEN Behavior_Segment_AsOf =
                         'SPARSE'
                    THEN 1 ELSE 0
                END
            )::BIGINT
                AS SPARSE,

            MAX(
                Feature_Cutoff_Year
            )::BIGINT
                AS Year

        FROM current_planning_inputs

        GROUP BY
            ABC_AsOf

        ORDER BY
            CASE ABC_AsOf
                WHEN 'A' THEN 1
                WHEN 'B' THEN 2
                WHEN 'C' THEN 3
                ELSE 4
            END
        """
    )



def get_inventory_demand_explorer() -> pd.DataFrame:

    return query_df(
        """
        WITH context AS (
            SELECT
                Historical_Start_Year,
                Latest_Completed_Year
            FROM current_system_context
        ),

        historical AS (
            SELECT
                f.Medicine_Key,

                COUNT(*)::BIGINT
                    AS Historical_Year_Count,

                SUM(
                    COALESCE(
                        f.Annual_Issuance,
                        0
                    )
                )
                    AS Historical_Total_Issuance,

                AVG(
                    f.Annual_Issuance
                )
                    AS Mean_Annual_Issuance,

                MEDIAN(
                    f.Annual_Issuance
                )
                    AS Median_Annual_Issuance,

                (
                    STDDEV_SAMP(
                        f.Annual_Issuance
                    ) FILTER (
                        WHERE f.Annual_Issuance > 0
                    )
                    /
                    NULLIF(
                        AVG(
                            f.Annual_Issuance
                        ) FILTER (
                            WHERE f.Annual_Issuance > 0
                        ),
                        0
                    )
                )
                    AS CV_Positive_Issuance,

                REGR_SLOPE(
                    f.Annual_Issuance,
                    f.Year
                )
                    AS Trend_Slope

            FROM fact_inventory_annual f
            CROSS JOIN context c

            WHERE f.Year BETWEEN
                c.Historical_Start_Year
                AND c.Latest_Completed_Year

            GROUP BY
                f.Medicine_Key
        ),

        latest_fact AS (
            SELECT
                f.Medicine_Key,
                f.Annual_Issuance
                    AS Latest_Annual_Issuance,
                f.Annual_Consumption_Value
                    AS Latest_ACV,
                f.ACV_Computable
                    AS Latest_ACV_Computable,
                f.Data_Quality_Status
                    AS Latest_Data_Quality_Status

            FROM fact_inventory_annual f

            WHERE f.Year = (
                SELECT Latest_Completed_Year
                FROM context
            )
        ),

        latest_abc AS (
            SELECT
                a.Medicine_Key,
                a.ABC_Eligibility

            FROM abc_classification_annual a

            WHERE a.Year = (
                SELECT Latest_Completed_Year
                FROM context
            )
        )

        SELECT
            p.Medicine_Key,
            p.Brand_Name,
            p.Generic_Name,
            p.Unit,

            p.ABC_AsOf
                AS ABC_Class,

            a.ABC_Eligibility,

            p.Behavior_Segment_AsOf
                AS Behavior_Segment,

            p.Persistence_AsOf
                AS Persistence,

            p.Positive_Years_AsOf
                AS Positive_Issuance_Years,

            p.Zero_Years_AsOf
                AS Zero_Issuance_Years,

            h.Historical_Year_Count,

            h.Historical_Total_Issuance,

            h.Mean_Annual_Issuance,
            h.Median_Annual_Issuance,
            h.CV_Positive_Issuance,
            h.Trend_Slope,

            f.Latest_Annual_Issuance,
            f.Latest_ACV,
            f.Latest_ACV_Computable,
            f.Latest_Data_Quality_Status,

        FROM current_planning_inputs p

        LEFT JOIN historical h
            ON p.Medicine_Key =
               h.Medicine_Key

        LEFT JOIN latest_fact f
            ON p.Medicine_Key =
               f.Medicine_Key

        LEFT JOIN latest_abc a
            ON p.Medicine_Key =
               a.Medicine_Key

        ORDER BY
            p.Brand_Name,
            p.Medicine_Key
        """
    )


def get_medicine_annual_history(
    medicine_key: str,
) -> pd.DataFrame:

    return query_df(
        """
        SELECT
            f.Year,
            f.Medicine_Key,
            f.Brand_Name,
            f.Generic_Name,
            f.Unit,

            f.Annual_Issuance,
            f.Average_cost,
            f.Annual_Consumption_Value,
            f.ACV_Computable,
            f.Data_Quality_Status,

            a.ABC_Class,
            a.ABC_Eligibility,

            b.Behavior_Segment,
            b.Persistence,
            b.Positive_Issuance_Years,
            b.CV_Positive_Issuance

        FROM fact_inventory_annual f

        LEFT JOIN abc_classification_annual a
            ON f.Medicine_Key = a.Medicine_Key
           AND f.Year = a.Year

        LEFT JOIN abc_behavior_annual b
            ON f.Medicine_Key = b.Medicine_Key
           AND f.Year = b.Year

        WHERE f.Medicine_Key = ?

        ORDER BY f.Year
        """,
        [medicine_key],
    )


# ============================================================
# FORECASTING & UNCERTAINTY
# ============================================================

FORECAST_EVIDENCE_TABLES = {
    "Point Forecast Backtest":
        "forecast_benchmark_metrics_overall",

    "All Model Comparison":
        "forecast_all_model_comparison",

    "Quantile Validation":
        "forecast_quantile_metrics_overall",

    "Occurrence Validation":
        "forecast_occurrence_metrics",

    "Bootstrap Pairwise Evidence":
        "forecast_bootstrap_pairwise",

    "Current Segment Summary":
        "current_forecast_segment_summary",

    "Model Selection Governance":
        "forecast_model_selection_governance",

    "Current Forecast Governance":
        "current_forecast_governance",

    "Legacy 2026 Point-Model Decision":
        "forecast_model_governance",
}


def get_forecast_summary() -> dict:

    row = query_df(
        """
        SELECT
            COUNT(*) AS Planning_Medicines,

            SUM(
                COALESCE(
                    Point_Forecast,
                    0
                )
            ) AS Point_Total,

            SUM(
                COALESCE(
                    Hurdle_Point_Forecast_Reference,
                    0
                )
            ) AS Hurdle_Total,

            SUM(
                COALESCE(P50, 0)
            ) AS P50_Total,

            SUM(
                COALESCE(P80, 0)
            ) AS P80_Total,

            SUM(
                COALESCE(P90, 0)
            ) AS P90_Total,

            AVG(
                Hurdle_Occurrence_Probability
            ) AS Mean_Occurrence_Probability,

            SUM(
                CASE
                    WHEN Quantile_Adjustment_Applied
                    THEN 1 ELSE 0
                END
            ) AS Quantile_Adjusted_Medicines,

            SUM(
                CASE
                    WHEN Flag_P80_Below_Point
                    THEN 1 ELSE 0
                END
            ) AS Raw_P80_Below_Point_Count,

            SUM(
                CASE
                    WHEN Flag_P90_Below_Point
                    THEN 1 ELSE 0
                END
            ) AS Raw_P90_Below_Point_Count

        FROM current_planning_inputs
        """
    ).iloc[0]

    return {
        "Planning_Medicines":
            int(row["Planning_Medicines"]),

        "Point_Total":
            float(row["Point_Total"]),

        "Hurdle_Total":
            float(row["Hurdle_Total"]),

        "P50_Total":
            float(row["P50_Total"]),

        "P80_Total":
            float(row["P80_Total"]),

        "P90_Total":
            float(row["P90_Total"]),

        "Mean_Occurrence_Probability":
            float(
                row[
                    "Mean_Occurrence_Probability"
                ]
            ),

        "Quantile_Adjusted_Medicines":
            int(
                row[
                    "Quantile_Adjusted_Medicines"
                ]
            ),

        "Raw_P80_Below_Point_Count":
            int(
                row[
                    "Raw_P80_Below_Point_Count"
                ]
            ),

        "Raw_P90_Below_Point_Count":
            int(
                row[
                    "Raw_P90_Below_Point_Count"
                ]
            ),
    }


def get_forecast_profile() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            Medicine_Key,
            Brand_Name,
            Generic_Name,
            Unit,

            ABC_AsOf,
            Behavior_Segment_AsOf,

            Lag1_Issuance,
            Lag2_Issuance,
            Moving_Average_2,

            Point_Forecast,
            Point_Forecast_Model,
            Point_Forecast_Status,

            Hurdle_Occurrence_Probability,
            Hurdle_Conditional_Positive_Quantity,
            Hurdle_Point_Forecast_Reference,
            Hurdle_Model_Status,

            Raw_P50,
            Raw_P80,
            Raw_P90,

            P50,
            P80,
            P90,

            Raw_Quantile_Crossing,
            Quantile_Adjustment_Applied,

            P50_Use_Status,
            P80_Use_Status,
            P90_Use_Status,

            Residual_Calibration_Applied,
            Uncertainty_Policy,

            Forecast_Version

        FROM current_planning_inputs

        ORDER BY
            Brand_Name,
            Medicine_Key
        """
    )


def get_forecast_by_behavior() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            Behavior_Segment_AsOf
                AS Behavior_Segment,

            COUNT(*) AS Medicines,

            SUM(
                COALESCE(
                    Point_Forecast,
                    0
                )
            ) AS Point_Forecast,

            SUM(
                COALESCE(
                    Hurdle_Point_Forecast_Reference,
                    0
                )
            ) AS Hurdle_Reference,

            SUM(
                COALESCE(P50, 0)
            ) AS P50,

            SUM(
                COALESCE(P80, 0)
            ) AS P80,

            SUM(
                COALESCE(P90, 0)
            ) AS P90,

            AVG(
                Hurdle_Occurrence_Probability
            ) AS Mean_Occurrence_Probability

        FROM current_planning_inputs

        GROUP BY
            Behavior_Segment_AsOf

        ORDER BY
            CASE Behavior_Segment_AsOf
                WHEN 'PERSISTENT_STABLE'
                    THEN 1
                WHEN 'PERSISTENT_VARIABLE'
                    THEN 2
                WHEN 'SPARSE'
                    THEN 3
                WHEN 'NON_MOVING'
                    THEN 4
                ELSE 5
            END
        """
    )


def get_forecast_by_abc() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            ABC_AsOf AS ABC_Class,

            COUNT(*) AS Medicines,

            SUM(
                COALESCE(
                    Point_Forecast,
                    0
                )
            ) AS Point_Forecast,

            SUM(
                COALESCE(P80, 0)
            ) AS P80,

            SUM(
                COALESCE(P90, 0)
            ) AS P90,

            AVG(
                Hurdle_Occurrence_Probability
            ) AS Mean_Occurrence_Probability

        FROM current_planning_inputs

        GROUP BY
            ABC_AsOf

        ORDER BY
            CASE ABC_AsOf
                WHEN 'A' THEN 1
                WHEN 'B' THEN 2
                WHEN 'C' THEN 3
                ELSE 4
            END
        """
    )


def get_forecast_evidence_table(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in FORECAST_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown forecast evidence table."
        )

    table = (
        FORECAST_EVIDENCE_TABLES[
            display_name
        ]
    )

    sql = (
        'SELECT * FROM "'
        + table
        + '"'
    )

    return query_df(sql)


# ============================================================
# PROCUREMENT DECISION SUPPORT
# ============================================================

PROCUREMENT_EVIDENCE_TABLES = {
    "Validated Budget Scenarios":
        "current_milp_budget_scenarios",

    "Budget Scenario Summary":
        "planning_2026_milp_budget_summary",

    "Budget Results by ABC":
        "planning_2026_milp_budget_by_abc",

    "Medicine Allocations":
        "planning_2026_milp_budget_allocations",

    "Strict Priority Audit":
        "current_milp_strict_priority_audit",

    "MILP Governance":
        "current_milp_governance",

    "Solver Governance":
        "current_milp_full_funding_governance",

    "Planning Governance":
        "current_planning_governance",
}


def get_procurement_readiness_summary() -> dict:

    row = query_df(
        """
        SELECT
            COUNT(*) AS Planning_Medicines,

            SUM(
                CASE
                    WHEN Optimizer_Core_Input_Ready
                    THEN 1 ELSE 0
                END
            ) AS Optimizer_Ready,

            SUM(
                CASE
                    WHEN NOT Optimizer_Core_Input_Ready
                    THEN 1 ELSE 0
                END
            ) AS Not_Optimizer_Ready,

            SUM(
                CASE
                    WHEN Needs_Current_Stock_Override
                    THEN 1 ELSE 0
                END
            ) AS Needs_Stock_Override,

            SUM(
                CASE
                    WHEN Needs_Planning_Cost_Override
                    THEN 1 ELSE 0
                END
            ) AS Needs_Cost_Override,

            SUM(
                COALESCE(
                    Net_Requirement_Point,
                    0
                )
            ) AS Point_Net_Requirement,

            SUM(
                COALESCE(
                    Net_Requirement_P80,
                    0
                )
            ) AS P80_Net_Requirement,

            SUM(
                COALESCE(
                    Net_Requirement_P90,
                    0
                )
            ) AS P90_Net_Requirement,

            SUM(
                COALESCE(
                    Estimated_Procurement_Value_Point,
                    0
                )
            ) AS Point_Value,

            SUM(
                COALESCE(
                    Estimated_Procurement_Value_P80,
                    0
                )
            ) AS P80_Value,

            SUM(
                COALESCE(
                    Estimated_Procurement_Value_P90,
                    0
                )
            ) AS P90_Value

        FROM current_planning_inputs
        """
    ).iloc[0]

    return {
        key: (
            int(value)
            if key in {
                "Planning_Medicines",
                "Optimizer_Ready",
                "Not_Optimizer_Ready",
                "Needs_Stock_Override",
                "Needs_Cost_Override",
            }
            else float(value)
        )
        for key, value
        in row.to_dict().items()
    }


def get_procurement_medicine_inputs() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            Medicine_Key,
            Brand_Name,
            Generic_Name,
            Unit,

            ABC_AsOf,
            Behavior_Segment_AsOf,

            Point_Forecast,
            P80,
            P90,

            Planning_Stock_Qty,
            Planning_Stock_Valid,
            Planning_Stock_Status,
            Planning_Stock_Source_Year,
            Planning_Stock_Source_Type,

            Planning_Cost,
            Planning_Cost_Available,
            Planning_Cost_Status,
            Planning_Cost_Source_Year,
            Planning_Cost_Age_Years,
            Planning_Cost_Source_Type,

            Requirement_Point,
            Requirement_P80,
            Requirement_P90,

            Net_Requirement_Point,
            Net_Requirement_P80,
            Net_Requirement_P90,

            Estimated_Procurement_Value_Point,
            Estimated_Procurement_Value_P80,
            Estimated_Procurement_Value_P90,

            Flag_P80_Below_Point,
            Flag_P90_Below_Point,

            Optimizer_Core_Input_Ready,
            Planning_Input_Status,

            Needs_Current_Stock_Override,
            Needs_Planning_Cost_Override,

            Planning_Version

        FROM current_planning_inputs

        ORDER BY
            CASE ABC_AsOf
                WHEN 'A' THEN 1
                WHEN 'B' THEN 2
                WHEN 'C' THEN 3
                ELSE 4
            END,
            Brand_Name,
            Medicine_Key
        """
    )


def get_procurement_evidence_table(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in PROCUREMENT_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown procurement evidence table."
        )

    table = (
        PROCUREMENT_EVIDENCE_TABLES[
            display_name
        ]
    )

    sql = (
        'SELECT * FROM "'
        + table
        + '"'
    )

    return query_df(sql)


def get_procurement_table_schema(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in PROCUREMENT_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown procurement evidence table."
        )

    table = (
        PROCUREMENT_EVIDENCE_TABLES[
            display_name
        ]
    )

    return query_df(
        """
        SELECT
            ordinal_position,
            column_name,
            data_type

        FROM information_schema.columns

        WHERE table_schema = 'main'
          AND table_name = ?

        ORDER BY ordinal_position
        """,
        [table],
    )


# ============================================================
# PROCUREMENT OPTIMIZER SCENARIO ACCESS
# ============================================================

def get_procurement_optimizer_options() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            Requirement_Scenario,
            Budget_Factor,
            Budget_Percent,
            MILP_B100,
            Scenario_Budget,
            Budget_Is_Actual_CDH_Budget,
            Budget_Definition,
            MILP_Input_Version

        FROM current_milp_budget_scenarios

        ORDER BY
            CASE Requirement_Scenario
                WHEN 'POINT' THEN 1
                WHEN 'P80_FLOORED_CEIL' THEN 2
                WHEN 'P90_FLOORED_CEIL' THEN 3
                ELSE 4
            END,
            Budget_Percent
        """
    )


def get_procurement_optimizer_result(
    requirement_scenario: str,
    budget_percent: float,
) -> dict:

    valid_scenarios = {
        "POINT",
        "P80_FLOORED_CEIL",
        "P90_FLOORED_CEIL",
    }

    valid_budgets = {
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
    }

    if requirement_scenario not in valid_scenarios:
        raise ValueError(
            "Unknown requirement scenario."
        )

    budget_percent = float(
        budget_percent
    )

    if budget_percent not in valid_budgets:
        raise ValueError(
            "Unsupported budget percentage."
        )

    scenario = query_df(
        """
        SELECT *
        FROM current_milp_budget_scenarios
        WHERE Requirement_Scenario = ?
          AND Budget_Percent = ?
        """,
        [
            requirement_scenario,
            budget_percent,
        ],
    )

    if len(scenario) != 1:
        raise ValueError(
            "Expected exactly one validated "
            "budget scenario."
        )

    # --------------------------------------------------------
    # 60?90%:
    # authoritative strict six-stage ABC optimizer V2
    # --------------------------------------------------------

    if budget_percent < 100.0:

        overview = query_df(
            """
            WITH selected AS (
                SELECT *
                FROM current_milp_strict_summary
                WHERE Requirement_Scenario = ?
                  AND Budget_Percent = ?
            )

            SELECT
                s.Requirement_Scenario,
                s.Budget_Factor,
                s.Budget_Percent,

                'STRICT_ABC_BUDGET_CONSTRAINED_MILP'
                    AS Result_Mode,

                (
                    SELECT COUNT(*)
                    FROM current_milp_strict_allocations a
                    WHERE
                        a.Requirement_Scenario =
                            s.Requirement_Scenario
                        AND a.Budget_Percent =
                            s.Budget_Percent
                )
                    AS Eligible_Medicine_Count,

                (
                    SELECT COUNT(*)
                    FROM current_milp_strict_allocations a
                    WHERE
                        a.Requirement_Scenario =
                            s.Requirement_Scenario
                        AND a.Budget_Percent =
                            s.Budget_Percent
                        AND a.Net_Requirement > 0
                )
                    AS Positive_Net_Requirement_Count,

                s.Required_Units,
                s.Allocated_Units,

                s.Unfunded_Units
                    AS Unfunded_Requirement_Units,

                s.B100,
                s.Scenario_Budget,

                s.B100
                    AS Required_Value,

                s.Allocated_Value,

                (
                    s.B100
                    - s.Allocated_Value
                )
                    AS Unfunded_Requirement_Value,

                s.Unspent_Budget,
                s.Budget_Utilization_Pct,

                s.Theta_A_Star
                    AS Theta_A,

                s.Theta_B_Star
                    AS Theta_B,

                s.Theta_C_Star
                    AS Theta_C,

                s.Optimization_Version

            FROM selected s
            """,
            [
                requirement_scenario,
                budget_percent,
            ],
        )

        by_abc = query_df(
            """
            SELECT *
            FROM current_milp_strict_by_abc
            WHERE Requirement_Scenario = ?
              AND Budget_Percent = ?
            ORDER BY
                CASE ABC_AsOf
                    WHEN 'A' THEN 1
                    WHEN 'B' THEN 2
                    WHEN 'C' THEN 3
                    ELSE 4
                END
            """,
            [
                requirement_scenario,
                budget_percent,
            ],
        )

        allocations = query_df(
            """
            SELECT *
            FROM current_milp_strict_allocations
            WHERE Requirement_Scenario = ?
              AND Budget_Percent = ?
            ORDER BY
                CASE ABC_AsOf
                    WHEN 'A' THEN 1
                    WHEN 'B' THEN 2
                    WHEN 'C' THEN 3
                    ELSE 4
                END,
                Brand_Name,
                Medicine_Key
            """,
            [
                requirement_scenario,
                budget_percent,
            ],
        )

    # --------------------------------------------------------
    # 100%:
    # separately validated exact full-funding result
    # --------------------------------------------------------

    else:

        overview = query_df(
            """
            SELECT
                Scenario
                    AS Requirement_Scenario,

                1.0
                    AS Budget_Factor,

                100.0
                    AS Budget_Percent,

                'FULL_FUNDING_VALIDATION'
                    AS Result_Mode,

                Eligible_Medicine_Count,

                Positive_Net_Requirement_Count,

                Total_Required_Units
                    AS Required_Units,

                Total_Allocated_Units
                    AS Allocated_Units,

                (
                    Total_Required_Units
                    - Total_Allocated_Units
                )
                    AS Unfunded_Requirement_Units,

                B100,

                B100
                    AS Scenario_Budget,

                Total_Required_Value
                    AS Required_Value,

                Total_Allocated_Value
                    AS Allocated_Value,

                (
                    Total_Required_Value
                    - Total_Allocated_Value
                )
                    AS Unfunded_Requirement_Value,

                Budget_Difference
                    AS Unspent_Budget,

                Budget_Utilization_Pct,

                Theta_A,
                Theta_B,
                Theta_C,

                Validation_VERSION
                    AS Optimization_Version

            FROM current_milp_full_funding_validation

            WHERE Scenario = ?
            """,
            [requirement_scenario],
        )

        by_abc = query_df(
            """
            SELECT *
            FROM current_milp_full_funding_by_abc
            WHERE Scenario = ?
            ORDER BY
                CASE ABC_AsOf
                    WHEN 'A' THEN 1
                    WHEN 'B' THEN 2
                    WHEN 'C' THEN 3
                    ELSE 4
                END
            """,
            [requirement_scenario],
        )

        allocations = query_df(
            """
            SELECT *
            FROM current_milp_full_funding_allocations
            WHERE Scenario = ?
            ORDER BY
                CASE ABC_AsOf
                    WHEN 'A' THEN 1
                    WHEN 'B' THEN 2
                    WHEN 'C' THEN 3
                    ELSE 4
                END,
                Brand_Name,
                Medicine_Key
            """,
            [requirement_scenario],
        )

    if len(overview) != 1:
        raise ValueError(
            "Expected exactly one optimizer "
            "overview row."
        )

    return {
        "scenario": scenario,
        "overview": overview,
        "by_abc": by_abc,
        "allocations": allocations,
    }


PROCUREMENT_EVIDENCE_TABLES.update(
    {
        "MILP Scenario Definition":
            "current_milp_budget_scenarios",

        "Full-Funding Validation":
            "current_milp_full_funding_validation",

        "Full-Funding by ABC":
            "current_milp_full_funding_by_abc",

        "Full-Funding Allocations":
            "current_milp_full_funding_allocations",
    }
)


# ============================================================
# RETROSPECTIVE VALIDATION
# ============================================================

RETROSPECTIVE_EVIDENCE_TABLES = {
    "Matched-Budget Summary":
        "retrospective_2025_matched_budget_summary",

    "Matched-Budget Trade-off":
        "retrospective_2025_matched_budget_tradeoff",

    "Results by ABC":
        "retrospective_2025_matched_budget_by_abc",

    "Alternate-Optimum Ranges":
        "retrospective_2025_matched_budget_altopt_ranges",

    "Policy Robustness":
        "retrospective_2025_matched_budget_altopt_policy_robustness",

    "Full Repeatability Audit":
        "retrospective_2025_matched_budget_full_repeatability_audit",

    "Repeatability Governance":
        "retrospective_2025_matched_budget_full_repeatability_governance",

    "Alternate-Optimum Governance":
        "retrospective_2025_matched_budget_altopt_governance",

    "Matched-Budget Governance":
        "retrospective_2025_matched_budget_governance",

    "Retrospective Governance":
        "retrospective_2025_governance",

    "Simulation Governance":
        "retrospective_2025_simulation_governance",
}


def get_retrospective_core_tables() -> dict:

    return {
        "summary": query_df(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_summary
            """
        ),

        "tradeoff": query_df(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_tradeoff
            """
        ),

        "altopt_ranges": query_df(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_altopt_ranges
            """
        ),

        "policy_robustness": query_df(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_altopt_policy_robustness
            """
        ),

        "repeatability": query_df(
            """
            SELECT *
            FROM retrospective_2025_matched_budget_full_repeatability_audit
            """
        ),
    }


def get_retrospective_evidence_table(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in RETROSPECTIVE_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown retrospective evidence table."
        )

    table = (
        RETROSPECTIVE_EVIDENCE_TABLES[
            display_name
        ]
    )

    sql = (
        'SELECT * FROM "'
        + table
        + '"'
    )

    return query_df(sql)


def get_retrospective_table_schema(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in RETROSPECTIVE_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown retrospective evidence table."
        )

    table = (
        RETROSPECTIVE_EVIDENCE_TABLES[
            display_name
        ]
    )

    return query_df(
        """
        SELECT
            ordinal_position,
            column_name,
            data_type

        FROM information_schema.columns

        WHERE table_schema = 'main'
          AND table_name = ?

        ORDER BY ordinal_position
        """,
        [table],
    )


# ============================================================
# EXPIRY & DATA QUALITY
# ============================================================

EXPIRY_EVIDENCE_TABLES = {
    "Medicine Expiry Summary":
        "expiry_dss_medicine_summary",

    "Expiry Identity Crosswalk":
        "expiry_medicine_identity_crosswalk",

    "Expiry Identity Review Queue":
        "expiry_medicine_review_queue",

    "Validated Expiry Records":
        "fact_expiry_medicine_records",

    "Expiry Validation Summary":
        "expiry_validation_summary",

    "Expiry DSS Governance":
        "expiry_dss_governance",

    "Expiry Source Governance":
        "expiry_source_governance",

    "Planning Review Queue":
        "current_review_queue",

    "Planning Review Summary":
        "current_review_summary",
}


def get_expiry_dss_summary() -> dict:

    row = query_df(
        """
        SELECT
            COUNT(*) AS Planning_Medicines,

            SUM(
                CASE
                    WHEN Has_Authoritative_Expiry_History
                    THEN 1 ELSE 0
                END
            ) AS With_Expiry_History,

            SUM(
                CASE
                    WHEN NOT Has_Authoritative_Expiry_History
                    THEN 1 ELSE 0
                END
            ) AS Without_Expiry_History

        FROM expiry_dss_medicine_summary
        """
    ).iloc[0]

    unresolved = scalar(
        """
        SELECT COUNT(*)
        FROM expiry_medicine_review_queue
        """
    )

    linked_records = scalar(
        """
        SELECT COUNT(*)
        FROM fact_expiry_medicine_records
        """
    )

    planning_review = scalar(
        """
        SELECT COUNT(*)
        FROM current_review_queue
        """
    )

    return {
        "Planning_Medicines":
            int(row["Planning_Medicines"]),

        "With_Expiry_History":
            int(row["With_Expiry_History"]),

        "Without_Expiry_History":
            int(row["Without_Expiry_History"]),

        "Unresolved_Expiry_Identities":
            int(unresolved),

        "Validated_Medicine_Domain_Records":
            int(linked_records),

        "Planning_Review_Rows":
            int(planning_review),
    }


def get_expiry_medicine_explorer() -> pd.DataFrame:

    return query_df(
        """
        SELECT
            d.Medicine_Key,
            d.Brand_Name,
            d.Generic_Name,
            d.Unit,

            e.* EXCLUDE (Medicine_Key)

        FROM dim_medicine d

        LEFT JOIN expiry_dss_medicine_summary e
            ON d.Medicine_Key = e.Medicine_Key

        ORDER BY
            d.Brand_Name,
            d.Medicine_Key
        """
    )


def get_expiry_fact_records() -> pd.DataFrame:

    return query_df(
        """
        SELECT *
        FROM fact_expiry_medicine_records
        """
    )


def get_expiry_records_for_medicine(
    medicine_key: str,
) -> pd.DataFrame:

    return query_df(
        """
        SELECT *
        FROM fact_expiry_medicine_records
        WHERE Medicine_Key = ?
        """,
        [medicine_key],
    )


def get_expiry_evidence_table(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in EXPIRY_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown expiry evidence table."
        )

    table = (
        EXPIRY_EVIDENCE_TABLES[
            display_name
        ]
    )

    sql = (
        'SELECT * FROM "'
        + table
        + '"'
    )

    return query_df(sql)


def get_expiry_table_schema(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in EXPIRY_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown expiry evidence table."
        )

    table = (
        EXPIRY_EVIDENCE_TABLES[
            display_name
        ]
    )

    return query_df(
        """
        SELECT
            ordinal_position,
            column_name,
            data_type

        FROM information_schema.columns

        WHERE table_schema = 'main'
          AND table_name = ?

        ORDER BY ordinal_position
        """,
        [table],
    )


# ============================================================
# GOVERNANCE & METHODOLOGY
# ============================================================

GOVERNANCE_EVIDENCE_TABLES = {
    "Forecast Model Governance":
        "forecast_model_selection_governance",

    "2026 Forecast Governance":
        "current_forecast_governance",

    "Planning Governance":
        "current_planning_governance",

    "MILP Governance":
        "current_milp_governance",

    "MILP Solver Governance":
        "current_milp_full_funding_governance",

    "MILP Formulation":
        "planning_2026_milp_formulation",

    "Strict Priority Governance":
        "current_milp_strict_governance",

    "Retrospective Governance":
        "retrospective_2025_governance",

    "Matched-Budget Governance":
        "retrospective_2025_matched_budget_governance",

    "Alternate-Optimum Governance":
        "retrospective_2025_matched_budget_altopt_governance",

    "Expiry Source Governance":
        "expiry_source_governance",

    "Expiry DSS Governance":
        "expiry_dss_governance",

    "ETL Run Audit":
        "etl_run_audit",

    "Forecast Feature Dictionary":
        "forecast_feature_dictionary",

    "Forecast Leakage Audit":
        "forecast_leakage_audit",
}


def get_governance_overview() -> dict:

    context = get_system_context()

    return {
        "Historical_Start_Year":
            context["Historical_Start_Year"],

        "Historical_End_Year":
            context["Latest_Completed_Year"],

        "Planning_Target_Year":
            context["Planning_Year"],

        "Medicine_Count": int(
            scalar(
                """
                SELECT COUNT(*)
                FROM dim_medicine
                """
            )
        ),

        "Annual_Inventory_Rows": int(
            scalar(
                """
                SELECT COUNT(*)
                FROM fact_inventory_annual
                """
            )
        ),

        "Optimizer_Ready": int(
            scalar(
                """
                SELECT COUNT(*)
                FROM current_planning_inputs
                WHERE Optimizer_Core_Input_Ready
                """
            )
        ),

        "Retrospective_Cases": int(
            scalar(
                """
                SELECT COUNT(*)
                FROM retrospective_2025_matched_budget_summary
                """
            )
        ),

        "Expiry_Linked_Medicines": int(
            scalar(
                """
                SELECT COUNT(*)
                FROM expiry_dss_medicine_summary
                WHERE Has_Authoritative_Expiry_History
                """
            )
        ),
    }


def get_governance_evidence_table(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in GOVERNANCE_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown governance evidence table."
        )

    table = (
        GOVERNANCE_EVIDENCE_TABLES[
            display_name
        ]
    )

    sql = (
        'SELECT * FROM "'
        + table
        + '"'
    )

    return query_df(sql)


def get_governance_table_schema(
    display_name: str,
) -> pd.DataFrame:

    if (
        display_name
        not in GOVERNANCE_EVIDENCE_TABLES
    ):
        raise ValueError(
            "Unknown governance evidence table."
        )

    table = (
        GOVERNANCE_EVIDENCE_TABLES[
            display_name
        ]
    )

    return query_df(
        """
        SELECT
            ordinal_position,
            column_name,
            data_type

        FROM information_schema.columns

        WHERE table_schema = 'main'
          AND table_name = ?

        ORDER BY ordinal_position
        """,
        [table],
    )


def get_governance_table_health() -> pd.DataFrame:

    rows = []

    connection = connect_read_only()

    try:

        existing = {
            row[0]
            for row in connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }

        for (
            display_name,
            table_name
        ) in GOVERNANCE_EVIDENCE_TABLES.items():

            available = (
                table_name in existing
            )

            if available:

                row_count = (
                    connection.execute(
                        f'SELECT COUNT(*) '
                        f'FROM "{table_name}"'
                    )
                    .fetchone()[0]
                )

            else:
                row_count = None

            rows.append(
                {
                    "Evidence": display_name,
                    "Table": table_name,
                    "Available": available,
                    "Rows": row_count,
                }
            )

    finally:
        connection.close()

    return pd.DataFrame(rows)



# ============================================================
# AUTHORITATIVE STRICT V2 DSS EVIDENCE MAPPING
# ============================================================

# The earlier budget-optimization V1 tables remain in DuckDB
# for analytical provenance. The prospective DSS uses strict
# six-stage ABC lexicographic optimization V2 for constrained
# 60?90% scenarios.

PROCUREMENT_EVIDENCE_TABLES.update(
    {
        "Budget Scenario Summary":
            "current_milp_strict_summary",

        "Budget Results by ABC":
            "current_milp_strict_by_abc",

        "Medicine Allocations":
            "current_milp_strict_allocations",

        "MILP Governance":
            "current_milp_strict_governance",
    }
)

# Do not present the earlier four-stage formulation as the
# final methodology in the primary Governance evidence menu.

GOVERNANCE_EVIDENCE_TABLES.pop(
    "MILP Governance",
    None,
)

GOVERNANCE_EVIDENCE_TABLES.pop(
    "MILP Formulation",
    None,
)

GOVERNANCE_EVIDENCE_TABLES.update(
    {
        "Strict Priority Audit":
            "current_milp_strict_priority_audit",

        "Prospective Strict Summary":
            "current_milp_strict_summary",
    }
)
