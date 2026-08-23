"""Construct in-memory one-step forecasting windows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def make_windows(
    sequence: Sequence[float] | np.ndarray, window_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping univariate inputs and one-step targets.

    The returned arrays have shapes ``(T - N, N, 1)`` and ``(T - N, 1)``.
    """
    if isinstance(window_length, bool) or not isinstance(window_length, (int, np.integer)):
        raise TypeError("window_length must be a positive integer")
    if window_length <= 0:
        raise ValueError("window_length must be positive")

    try:
        values = np.asarray(sequence, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("sequence must contain numeric values") from error

    if values.ndim != 1:
        raise ValueError("sequence must be one-dimensional")
    if values.size <= window_length:
        raise ValueError("sequence must contain more than window_length observations")
    if not np.all(np.isfinite(values)):
        raise ValueError("sequence must contain only finite values")

    num_windows = values.size - window_length
    inputs = np.empty((num_windows, window_length, 1), dtype=float)
    targets = np.empty((num_windows, 1), dtype=float)
    for index in range(num_windows):
        inputs[index, :, 0] = values[index : index + window_length]
        targets[index, 0] = values[index + window_length]
    return inputs, targets


def load_value_csv(path: str | Path) -> np.ndarray:
    """Load the ``value`` column from a ``t,value`` CSV file."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["t", "value"]:
            raise ValueError("CSV must have exactly the columns 't' and 'value'")
        try:
            return np.asarray([row["value"] for row in reader], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("CSV value column must contain numeric values") from error
