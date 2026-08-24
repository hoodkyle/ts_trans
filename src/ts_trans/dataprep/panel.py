"""Validate long-form panel data and construct per-series windows."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .windows import make_windows


_FREQUENCIES = {"monthly": "M", "quarterly": "Q"}
_PERIOD_COLUMN = "__ts_trans_period"
_SERIAL_COLUMN = "__ts_trans_time_serial"


def _as_cross_section_cols(cross_section_cols: Iterable[str]) -> list[str]:
    if isinstance(cross_section_cols, str):
        cross_section_cols = [cross_section_cols]
    columns = list(cross_section_cols)
    if not columns:
        raise ValueError("cross_section_cols must contain at least one column")
    if len(set(columns)) != len(columns):
        raise ValueError("cross_section_cols must not contain duplicate columns")
    return columns


def _periods(time_values: pd.Series, frequency: str) -> pd.Series:
    if frequency not in _FREQUENCIES:
        raise ValueError("frequency must be 'monthly' or 'quarterly'")
    timestamps = pd.to_datetime(time_values, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("time column must contain valid, non-missing datetimes")
    return timestamps.dt.to_period(_FREQUENCIES[frequency])


def prepare_panel(
    df: pd.DataFrame,
    cross_section_cols: Iterable[str],
    time_col: str,
    value_col: str,
    frequency: str,
) -> pd.DataFrame:
    """Validate and return a sorted, canonical copy of long-form panel data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    cross_section_cols = _as_cross_section_cols(cross_section_cols)
    if len(set(cross_section_cols + [time_col, value_col])) != len(cross_section_cols) + 2:
        raise ValueError("cross_section_cols, time_col, and value_col must be distinct columns")
    if df.empty:
        raise ValueError("df must contain at least one row")
    required = cross_section_cols + [time_col, value_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if _PERIOD_COLUMN in df.columns or _SERIAL_COLUMN in df.columns:
        raise ValueError("input uses reserved internal time column names")

    prepared = df.copy(deep=True)
    if prepared[cross_section_cols].isna().any().any():
        raise ValueError("cross-section keys must not be missing")

    prepared[_PERIOD_COLUMN] = _periods(prepared[time_col], frequency)
    if prepared.duplicated(cross_section_cols + [_PERIOD_COLUMN]).any():
        raise ValueError("duplicate observation for a cross-section and time period")

    values = pd.to_numeric(prepared[value_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("value column must contain finite numeric values")
    prepared[value_col] = values.astype(float)
    prepared[_SERIAL_COLUMN] = prepared[_PERIOD_COLUMN].astype("int64")

    sort_columns = cross_section_cols + [_SERIAL_COLUMN]
    prepared = prepared.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    for _, group in prepared.groupby(cross_section_cols, sort=False, dropna=False):
        serial = group[_SERIAL_COLUMN].to_numpy()
        if len(serial) > 1 and not np.all(np.diff(serial) == 1):
            raise ValueError("time series contains a missing period")
    return prepared


def make_panel_windows(
    df: pd.DataFrame,
    cross_section_cols: Iterable[str],
    time_col: str,
    value_col: str,
    frequency: str,
    window_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct one-step windows independently for each panel series."""
    cross_section_cols = _as_cross_section_cols(cross_section_cols)
    prepared = prepare_panel(df, cross_section_cols, time_col, value_col, frequency)
    input_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    metadata_parts: list[dict[str, object]] = []

    for _, group in prepared.groupby(cross_section_cols, sort=False, dropna=False):
        inputs, targets = make_windows(group[value_col].to_numpy(), window_length)
        input_parts.append(inputs)
        target_parts.append(targets)
        times = group[time_col].reset_index(drop=True)
        for index in range(len(targets)):
            row = {column: group.iloc[0][column] for column in cross_section_cols}
            row.update(
                {
                    "forecast_time": times.iloc[index + window_length],
                    "window_start_time": times.iloc[index],
                    "window_end_time": times.iloc[index + window_length - 1],
                }
            )
            metadata_parts.append(row)

    metadata = pd.DataFrame(metadata_parts)
    return np.concatenate(input_parts), np.concatenate(target_parts), metadata
