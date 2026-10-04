from __future__ import annotations

from dataclasses import dataclass

try:
    from .runtime_context import (
        RuntimeContext,
        load_runtime_context_from_env,
    )
except ImportError:
    from runtime_context import (
        RuntimeContext,
        load_runtime_context_from_env,
    )


@dataclass(frozen=True)
class ForecastEvaluationContext:
    supervised_target_years: list[int]
    folds: list[dict]
    combined_oos_label: str
    evaluation_populations: list[str]


def build_forecast_evaluation_context(
    runtime_context: RuntimeContext | None = None,
) -> ForecastEvaluationContext:

    context = (
        load_runtime_context_from_env()
        if runtime_context is None
        else runtime_context
    )

    supervised_years = (
        context.supervised_target_years
    )

    if len(supervised_years) < 3:
        raise ValueError(
            "At least three supervised target years "
            "are required for chronological backtesting."
        )

    first_train_year = (
        supervised_years[0]
    )

    first_test_year = (
        first_train_year + 2
    )

    test_years = [
        year
        for year in supervised_years
        if year >= first_test_year
    ]

    folds = [
        {
            "Fold_Name":
                f"BACKTEST_{test_year}",

            "Train_Start":
                first_train_year,

            "Train_End":
                test_year - 1,

            "Test_Year":
                test_year,
        }
        for test_year in test_years
    ]

    combined_oos_label = (
        f"COMBINED_OOS_"
        f"{test_years[0]}_"
        f"{test_years[-1]}"
    )

    evaluation_populations = (
        [
            fold["Fold_Name"]
            for fold in folds
        ]
        +
        [
            combined_oos_label
        ]
    )

    return ForecastEvaluationContext(
        supervised_target_years=(
            supervised_years
        ),
        folds=folds,
        combined_oos_label=(
            combined_oos_label
        ),
        evaluation_populations=(
            evaluation_populations
        ),
    )
