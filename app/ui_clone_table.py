"""Presentation-only, sortable tables for the CDH workspace UI.

All records come from the caller. Sorting and paging happen in the browser and
never write to the analytical database or change the caller's DataFrame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from streamlit.components.v2 import component


_ASSETS = Path(__file__).resolve().parent
_TABLE = component(
    "cdh_clone_table",
    css=(_ASSETS / "ui_clone_table.css").read_text(encoding="utf-8"),
    js=(_ASSETS / "ui_clone_table.js").read_text(encoding="utf-8"),
    isolate_styles=True,
)


def clone_table(
    df: pd.DataFrame,
    key: str,
    labels: Mapping[str, str] | None = None,
    page_size: int = 8,
):
    """Display live records in a sortable, paginated reference-style table.

    ``labels`` maps source column names to display labels. An optional boolean
    ``Selected`` column highlights its rows and is not displayed as a field.
    All other source columns remain available, including unknown status values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clone_table requires a pandas DataFrame")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise ValueError("page_size must be a positive integer")
    labels = labels or {}
    columns = []
    for position, column in enumerate(df.columns):
        series = df.iloc[:, position]
        name = str(column)
        if name == "Selected" and is_bool_dtype(series.dtype):
            kind = "selected"
        elif is_bool_dtype(series.dtype):
            kind = "boolean"
        elif is_numeric_dtype(series.dtype):
            kind = "number"
        elif is_datetime64_any_dtype(series.dtype):
            kind = "date"
        else:
            kind = "text"
        columns.append(
            {
                "name": name,
                "label": str(labels.get(column, name.replace("_", " "))),
                "kind": kind,
            }
        )

    # Pandas serializes nullable/numpy numbers, timestamps and NaN consistently.
    # The browser renders every value using textContent, never HTML templates.
    payload = json.loads(df.to_json(orient="split", date_format="iso", index=False))
    return _TABLE(
        key=key,
        data={
            "columns": columns,
            "rows": payload["data"],
            "pageSize": page_size,
            "label": str(key).replace("_", " ").replace("-", " "),
        },
    )
