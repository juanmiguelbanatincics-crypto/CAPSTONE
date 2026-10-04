from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    from .runtime_context import (
        load_runtime_context_from_env,
    )
    from .forecast_evaluation_context import (
        build_forecast_evaluation_context,
    )
except ImportError:
    from runtime_context import (
        load_runtime_context_from_env,
    )
    from forecast_evaluation_context import (
        build_forecast_evaluation_context,
    )


# ============================================================
# GOVERNED RUNTIME
# ============================================================

RUNTIME_CONTEXT = (
    load_runtime_context_from_env()
)

EVALUATION_CONTEXT = (
    build_forecast_evaluation_context(
        RUNTIME_CONTEXT
    )
)

DATABASE_PATH = (
    RUNTIME_CONTEXT.database_path
)

OUTPUT_DIR = (
    RUNTIME_CONTEXT.processed_dir
    / "forecasting"
    / "model_governance_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

POLICY_VERSION = (
    "FORECAST_MODEL_SELECTION_GOVERNANCE_V1"
)

COMBINED_OOS_LABEL = (
    EVALUATION_CONTEXT.combined_oos_label
)

LATEST_FOLD = (
    EVALUATION_CONTEXT.folds[-1][
        "Fold_Name"
    ]
)


# ============================================================
# TABLE HELPERS
# ============================================================

def table_exists(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> bool:

    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?
        """,
        [table_name],
    ).fetchone()[0]

    return bool(result)


# ============================================================
# RESOLVE PUBLISHED INCUMBENT
# ============================================================

def resolve_incumbent(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[str, str]:

    env_incumbent = (
        os.getenv(
            "CDH_FORECAST_INCUMBENT_MODEL"
        )
    )

    if env_incumbent:

        return (
            env_incumbent.strip(),
            "ENVIRONMENT_OVERRIDE",
        )

    if table_exists(
        connection,
        "forecast_model_selection_governance_history",
    ):

        prior = connection.execute(
            """
            SELECT Recommended_Model
            FROM forecast_model_selection_governance_history
            WHERE Publication_Status = 'PUBLISHED'
            ORDER BY Planning_Year DESC, Run_ID DESC
            LIMIT 1
            """
        ).fetchone()

        if prior is not None:

            return (
                str(prior[0]),
                "PRIOR_PUBLISHED_MODEL_GOVERNANCE_HISTORY",
            )

    if table_exists(
        connection,
        "forecast_model_selection_governance",
    ):

        prior = connection.execute(
            """
            SELECT Recommended_Model
            FROM forecast_model_selection_governance
            WHERE Publication_Status = 'PUBLISHED'
            ORDER BY Planning_Year DESC
            LIMIT 1
            """
        ).fetchone()

        if prior is not None:

            return (
                str(prior[0]),
                "PRIOR_PUBLISHED_MODEL_GOVERNANCE",
            )

    if table_exists(
        connection,
        "forecast_2026_governance",
    ):

        legacy = connection.execute(
            """
            SELECT Point_Forecast_Incumbent
            FROM forecast_2026_governance
            LIMIT 1
            """
        ).fetchone()

        if legacy is not None:

            return (
                str(legacy[0]),
                "LEGACY_2026_GOVERNANCE_SEED",
            )

    raise ValueError(
        "No published incumbent model could "
        "be resolved. Supply "
        "CDH_FORECAST_INCUMBENT_MODEL or "
        "provide prior published governance."
    )


# ============================================================
# LOAD EVIDENCE
# ============================================================

def load_evidence() -> tuple[
    str,
    str,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:

        incumbent, source = (
            resolve_incumbent(
                connection
            )
        )

        summary = connection.execute(
            """
            SELECT *
            FROM forecast_bootstrap_model_summary
            WHERE Evaluation_Population = ?
            ORDER BY Observed_WAPE_Pct, Model
            """,
            [COMBINED_OOS_LABEL],
        ).df()

        pairwise = connection.execute(
            """
            SELECT *
            FROM forecast_bootstrap_pairwise
            WHERE Evaluation_Population = ?
            """,
            [COMBINED_OOS_LABEL],
        ).df()

        fold_metrics = connection.execute(
            """
            SELECT *
            FROM forecast_all_model_comparison
            WHERE Fold_Name = ?
            """,
            [LATEST_FOLD],
        ).df()

    finally:

        connection.close()

    return (
        incumbent,
        source,
        summary,
        pairwise,
        fold_metrics,
    )


# ============================================================
# NORMALIZED PAIRWISE DIFFERENCE
# ============================================================

def get_pairwise_difference(
    pairwise: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
) -> dict:

    direct = pairwise.loc[
        pairwise[
            "Metric"
        ].eq(metric)
        &
        pairwise[
            "Model_A"
        ].eq(model_a)
        &
        pairwise[
            "Model_B"
        ].eq(model_b)
    ]

    reverse = pairwise.loc[
        pairwise[
            "Metric"
        ].eq(metric)
        &
        pairwise[
            "Model_A"
        ].eq(model_b)
        &
        pairwise[
            "Model_B"
        ].eq(model_a)
    ]

    if len(direct) == 1:

        row = direct.iloc[0]

        return {
            "observed":
                float(
                    row[
                        "Observed_Difference_A_minus_B"
                    ]
                ),

            "lower95":
                float(
                    row[
                        "Difference_Lower95"
                    ]
                ),

            "upper95":
                float(
                    row[
                        "Difference_Upper95"
                    ]
                ),
        }

    if len(reverse) == 1:

        row = reverse.iloc[0]

        return {
            "observed":
                -float(
                    row[
                        "Observed_Difference_A_minus_B"
                    ]
                ),

            "lower95":
                -float(
                    row[
                        "Difference_Upper95"
                    ]
                ),

            "upper95":
                -float(
                    row[
                        "Difference_Lower95"
                    ]
                ),
        }

    raise ValueError(
        f"Expected exactly one paired "
        f"comparison for {model_a} vs "
        f"{model_b} / {metric}."
    )


# ============================================================
# SELECT STRONGEST CHALLENGER
# ============================================================

def select_challenger(
    summary: pd.DataFrame,
    incumbent: str,
) -> str:

    if summary.empty:

        raise ValueError(
            "Combined bootstrap summary "
            "is empty."
        )

    models = set(
        summary[
            "Model"
        ]
        .tolist()
    )

    if incumbent not in models:

        raise ValueError(
            f"Published incumbent "
            f"{incumbent} is absent from "
            "the evaluated model set."
        )

    challengers = (
        summary.loc[
            ~summary[
                "Model"
            ].eq(
                incumbent
            )
        ]
        .sort_values(
            [
                "Observed_WAPE_Pct",
                "Model",
            ]
        )
    )

    if challengers.empty:

        raise ValueError(
            "No challenger model available."
        )

    return str(
        challengers.iloc[0][
            "Model"
        ]
    )


# ============================================================
# GOVERNANCE DECISION
# ============================================================

def build_decision(
    incumbent: str,
    incumbent_source: str,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:

    challenger = select_challenger(
        summary,
        incumbent,
    )

    incumbent_summary = (
        summary.loc[
            summary[
                "Model"
            ].eq(
                incumbent
            )
        ]
        .iloc[0]
    )

    challenger_summary = (
        summary.loc[
            summary[
                "Model"
            ].eq(
                challenger
            )
        ]
        .iloc[0]
    )

    wape_pair = (
        get_pairwise_difference(
            pairwise,
            challenger,
            incumbent,
            "WAPE_Pct",
        )
    )

    rmse_pair = (
        get_pairwise_difference(
            pairwise,
            challenger,
            incumbent,
            "RMSE",
        )
    )

    bias_pair = (
        get_pairwise_difference(
            pairwise,
            challenger,
            incumbent,
            "Absolute_Bias_Pct",
        )
    )

    incumbent_latest = (
        fold_metrics.loc[
            fold_metrics[
                "Model"
            ].eq(
                incumbent
            )
        ]
    )

    challenger_latest = (
        fold_metrics.loc[
            fold_metrics[
                "Model"
            ].eq(
                challenger
            )
        ]
    )

    if (
        len(incumbent_latest) != 1
        or
        len(challenger_latest) != 1
    ):

        raise ValueError(
            "Latest chronological fold "
            "metrics are incomplete."
        )

    incumbent_latest_wape = float(
        incumbent_latest.iloc[0][
            "WAPE_Pct"
        ]
    )

    challenger_latest_wape = float(
        challenger_latest.iloc[0][
            "WAPE_Pct"
        ]
    )

    incumbent_combined_wape = float(
        incumbent_summary[
            "Observed_WAPE_Pct"
        ]
    )

    challenger_combined_wape = float(
        challenger_summary[
            "Observed_WAPE_Pct"
        ]
    )

    observed_wape_better = (
        challenger_combined_wape
        <
        incumbent_combined_wape
    )

    stable_wape_improvement = (
        wape_pair[
            "upper95"
        ]
        < 0
    )

    latest_fold_better = (
        challenger_latest_wape
        <
        incumbent_latest_wape
    )

    rmse_stably_worse = (
        rmse_pair[
            "lower95"
        ]
        > 0
    )

    absolute_bias_stably_worse = (
        bias_pair[
            "lower95"
        ]
        > 0
    )

    if not observed_wape_better:

        decision = (
            "RETAIN_INCUMBENT"
        )

        reason = (
            "INCUMBENT_HAS_LOWER_OR_EQUAL_"
            "COMBINED_WAPE"
        )

    elif not stable_wape_improvement:

        decision = (
            "RETAIN_INCUMBENT"
        )

        reason = (
            "CHALLENGER_WAPE_ADVANTAGE_"
            "NOT_STABLE_IN_PAIRED_BOOTSTRAP"
        )

    elif not latest_fold_better:

        decision = (
            "RETAIN_INCUMBENT"
        )

        reason = (
            "CHALLENGER_NOT_BETTER_ON_"
            "LATEST_CHRONOLOGICAL_HOLDOUT"
        )

    elif rmse_stably_worse:

        decision = (
            "RETAIN_INCUMBENT"
        )

        reason = (
            "CHALLENGER_STABLY_WORSE_RMSE"
        )

    elif absolute_bias_stably_worse:

        decision = (
            "RETAIN_INCUMBENT"
        )

        reason = (
            "CHALLENGER_STABLY_WORSE_"
            "ABSOLUTE_BIAS"
        )

    else:

        decision = (
            "PROMOTE_CHALLENGER"
        )

        reason = (
            "CHALLENGER_PASSED_WAPE_"
            "STABILITY_RECENCY_AND_"
            "ERROR_GUARDRAILS"
        )

    recommended_model = (
        challenger
        if decision
        == "PROMOTE_CHALLENGER"
        else incumbent
    )

    if (
        RUNTIME_CONTEXT.run_mode
        == "TEST"
    ):

        publication_status = (
            "TEST_ONLY_NOT_PUBLISHABLE"
        )

    elif (
        RUNTIME_CONTEXT.run_mode
        == "STAGING"
    ):

        publication_status = (
            "STAGING_RECOMMENDATION_"
            "PENDING_PUBLICATION"
        )

    else:

        publication_status = (
            "EVALUATION_NOT_AUTO_PUBLISHED"
        )

    result = pd.DataFrame(
        [
            {
                "Policy_Version":
                    POLICY_VERSION,

                "Run_Mode":
                    RUNTIME_CONTEXT.run_mode,

                "Run_ID":
                    RUNTIME_CONTEXT.run_id,

                "Planning_Year":
                    RUNTIME_CONTEXT.planning_year,

                "Latest_Completed_Year":
                    RUNTIME_CONTEXT.latest_completed_year,

                "Evaluation_Population":
                    COMBINED_OOS_LABEL,

                "Latest_Backtest_Fold":
                    LATEST_FOLD,

                "Incumbent_Source":
                    incumbent_source,

                "Incumbent_Model":
                    incumbent,

                "Challenger_Model":
                    challenger,

                "Incumbent_Combined_WAPE_Pct":
                    incumbent_combined_wape,

                "Challenger_Combined_WAPE_Pct":
                    challenger_combined_wape,

                "Challenger_Minus_Incumbent_WAPE_Pct":
                    wape_pair[
                        "observed"
                    ],

                "WAPE_Difference_Lower95":
                    wape_pair[
                        "lower95"
                    ],

                "WAPE_Difference_Upper95":
                    wape_pair[
                        "upper95"
                    ],

                "Observed_WAPE_Better":
                    observed_wape_better,

                "Stable_WAPE_Improvement":
                    stable_wape_improvement,

                "Incumbent_Latest_Fold_WAPE_Pct":
                    incumbent_latest_wape,

                "Challenger_Latest_Fold_WAPE_Pct":
                    challenger_latest_wape,

                "Latest_Fold_WAPE_Better":
                    latest_fold_better,

                "Challenger_Minus_Incumbent_RMSE":
                    rmse_pair[
                        "observed"
                    ],

                "RMSE_Difference_Lower95":
                    rmse_pair[
                        "lower95"
                    ],

                "RMSE_Difference_Upper95":
                    rmse_pair[
                        "upper95"
                    ],

                "Challenger_Stably_Worse_RMSE":
                    rmse_stably_worse,

                "Challenger_Minus_Incumbent_Absolute_Bias_Pct":
                    bias_pair[
                        "observed"
                    ],

                "Absolute_Bias_Difference_Lower95":
                    bias_pair[
                        "lower95"
                    ],

                "Absolute_Bias_Difference_Upper95":
                    bias_pair[
                        "upper95"
                    ],

                "Challenger_Stably_Worse_Absolute_Bias":
                    absolute_bias_stably_worse,

                "Decision":
                    decision,

                "Decision_Reason":
                    reason,

                "Recommended_Model":
                    recommended_model,

                "Automatic_Promotion_Allowed":
                    False,

                "Publication_Status":
                    publication_status,
            }
        ]
    )

    return result


# ============================================================
# WRITE
# ============================================================

def write_outputs(
    governance: pd.DataFrame,
) -> None:

    governance.to_csv(
        OUTPUT_DIR
        / (
            "forecast_model_selection_"
            f"governance_{RUNTIME_CONTEXT.planning_year}.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:

        connection.register(
            "governance_df",
            governance,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
            forecast_model_selection_governance AS
            SELECT *
            FROM governance_df
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
            forecast_model_selection_governance_history
            AS
            SELECT *
            FROM governance_df
            WHERE 1 = 0
            """
        )

        connection.execute(
            """
            DELETE FROM
            forecast_model_selection_governance_history
            WHERE Run_ID = ?
            """,
            [
                RUNTIME_CONTEXT.run_id
            ],
        )

        connection.execute(
            """
            INSERT INTO
            forecast_model_selection_governance_history
            SELECT *
            FROM governance_df
            """
        )

    finally:

        connection.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 108)
    print("CDH INVENTORY DSS")
    print(
        "GOVERNED FORECAST MODEL "
        "SELECTION"
    )
    print("=" * 108)
    print()

    (
        incumbent,
        incumbent_source,
        summary,
        pairwise,
        fold_metrics,
    ) = load_evidence()

    governance = build_decision(
        incumbent=incumbent,
        incumbent_source=incumbent_source,
        summary=summary,
        pairwise=pairwise,
        fold_metrics=fold_metrics,
    )

    row = governance.iloc[0]

    print(
        "Evaluation population:",
        row[
            "Evaluation_Population"
        ],
    )

    print(
        "Published incumbent:",
        row[
            "Incumbent_Model"
        ],
    )

    print(
        "Strongest challenger:",
        row[
            "Challenger_Model"
        ],
    )

    print(
        "Decision:",
        row[
            "Decision"
        ],
    )

    print(
        "Reason:",
        row[
            "Decision_Reason"
        ],
    )

    print(
        "Recommended model:",
        row[
            "Recommended_Model"
        ],
    )

    print(
        "Publication status:",
        row[
            "Publication_Status"
        ],
    )

    print()

    write_outputs(
        governance
    )

    print(
        "Model-governance table written."
    )

    print(
        "No model was automatically "
        "published."
    )

    print("=" * 108)


if __name__ == "__main__":
    main()
