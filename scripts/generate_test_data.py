"""Generate small, deterministic univariate time-series CSV fixtures."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path


SERIES_LENGTHS = {
    "linear_trend": 120,
    "sine_wave": 160,
    "ar1_recursive": 140,
    "trend_plus_sine": 180,
}


def generate_series(name: str, length: int) -> list[float]:
    """Return one deterministic series identified by ``name``."""
    if name == "linear_trend":
        return [1.0 + 0.05 * t for t in range(length)]
    if name == "sine_wave":
        return [math.sin(2.0 * math.pi * t / 20.0) for t in range(length)]
    if name == "ar1_recursive":
        rng = random.Random(20260823)
        values = [0.0]
        for _ in range(1, length):
            values.append(0.8 * values[-1] + rng.uniform(-0.2, 0.2))
        return values
    if name == "trend_plus_sine":
        return [0.02 * t + 0.75 * math.sin(2.0 * math.pi * t / 30.0) for t in range(length)]
    raise ValueError(f"unknown series: {name}")


def write_series(path: Path, values: list[float]) -> None:
    """Write values as a simple ``t,value`` CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "value"])
        writer.writerows((t, f"{value:.10f}") for t, value in enumerate(values))


def generate_test_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, length in SERIES_LENGTHS.items():
        write_series(output_dir / f"{name}.csv", generate_series(name, length))


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    generate_test_data(repository_root / "data" / "testdata")
