"""Reversible per-series standardization for long-form panel data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


_MIN_STD = 1e-12


def _cross_section_cols(cross_section_cols: Iterable[str]) -> list[str]:
    if isinstance(cross_section_cols, str):
        cross_section_cols = [cross_section_cols]
    columns = list(cross_section_cols)
    if not columns:
        raise ValueError("cross_section_cols must contain at least one column")
    if len(set(columns)) != len(columns):
        raise ValueError("cross_section_cols must not contain duplicate columns")
    return columns


def standardize_panel(
    df: pd.DataFrame,
    cross_section_cols: Iterable[str],
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize values independently within each cross-sectional series.

    The population standard deviation (``ddof=0``) is used. Series with a
    standard deviation at or below ``1e-12`` are rejected as degenerate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    cross_section_cols = _cross_section_cols(cross_section_cols)
    required = cross_section_cols + [value_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if value_col in cross_section_cols:
        raise ValueError("cross_section_cols and value_col must be distinct columns")
    if df.empty:
        raise ValueError("df must contain at least one row")
    if df[cross_section_cols].isna().any().any():
        raise ValueError("cross-section keys must not be missing")

    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("value column must contain finite numeric values")

    scaled = df.copy(deep=True)
    scaled_values = np.empty_like(values)
    parameter_rows: list[dict[str, object]] = []
    grouped = df.groupby(cross_section_cols, sort=True, dropna=False)
    for key_values, positions in grouped.indices.items():
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        panel_values = values[positions]
        mean = float(panel_values.mean())
        std = float(panel_values.std(ddof=0))
        if not np.isfinite(std) or std <= _MIN_STD:
            raise ValueError("cross-sectional series has zero or near-zero standard deviation")
        scaled_values[positions] = (panel_values - mean) / std
        parameter_rows.append(dict(zip(cross_section_cols, key_values)) | {"mean": mean, "std": std})

    scaled[value_col] = scaled_values
    scaling_params = pd.DataFrame(parameter_rows, columns=cross_section_cols + ["mean", "std"])
    return scaled, scaling_params.sort_values(cross_section_cols, kind="mergesort").reset_index(drop=True)


def inverse_standardize(
    values: np.ndarray,
    metadata: pd.DataFrame,
    scaling_params: pd.DataFrame,
    cross_section_cols: Iterable[str],
) -> np.ndarray:
    """Return standardized predictions in their original panel units."""
    cross_section_cols = _cross_section_cols(cross_section_cols)
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        row_count = array.shape[0]
    elif array.ndim == 1:
        row_count = array.shape[0]
    else:
        raise ValueError("values must have shape (n,) or (n, 1)")
    if not isinstance(metadata, pd.DataFrame) or not isinstance(scaling_params, pd.DataFrame):
        raise TypeError("metadata and scaling_params must be pandas DataFrames")
    if len(metadata) != row_count:
        raise ValueError("number of values and metadata rows must match")
    missing_metadata = [column for column in cross_section_cols if column not in metadata.columns]
    missing_params = [column for column in cross_section_cols + ["mean", "std"] if column not in scaling_params.columns]
    if missing_metadata:
        raise ValueError(f"metadata is missing cross-section columns: {missing_metadata}")
    if missing_params:
        raise ValueError(f"scaling_params is missing columns: {missing_params}")
    if metadata[cross_section_cols].isna().any().any():
        raise ValueError("metadata cross-section keys must not be missing")
    if scaling_params[cross_section_cols].isna().any().any():
        raise ValueError("scaling_params cross-section keys must not be missing")
    if scaling_params.duplicated(cross_section_cols).any():
        raise ValueError("scaling_params must contain one row per cross-sectional series")

    matched = metadata[cross_section_cols].merge(
        scaling_params[cross_section_cols + ["mean", "std"]],
        on=cross_section_cols,
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
    )
    if (matched["_merge"] != "both").any():
        raise ValueError("metadata contains a cross-section with no scaling parameters")
    if not np.isfinite(matched[["mean", "std"]].to_numpy(dtype=float)).all() or (matched["std"] <= _MIN_STD).any():
        raise ValueError("scaling_params must contain finite, positive standard deviations")

    std = matched["std"].to_numpy()
    mean = matched["mean"].to_numpy()
    if array.ndim == 2:
        std = std[:, None]
        mean = mean[:, None]
    restored = array * std + mean
    return restored.reshape(array.shape)
