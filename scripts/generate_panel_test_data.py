"""Generate deterministic monthly panel fixtures for dataprep tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ts_trans.dataprep import make_panel_windows, prepare_panel


SEED = 20260824
N_PANELS = 100
N_OBSERVATIONS = 240
START_DATE = "2000-01"
WINDOW_LENGTH = 12
PANEL_IDS = [f"P{i:03d}" for i in range(N_PANELS)]


def _panel_frame(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel": np.repeat(PANEL_IDS, N_OBSERVATIONS),
            "date": np.tile(dates.strftime("%Y-%m"), N_PANELS),
            "value": values.reshape(-1),
        }
    )


def generate_linear(rng: np.random.Generator, dates: pd.DatetimeIndex):
    intercepts = rng.uniform(-2.0, 2.0, N_PANELS)
    slopes = rng.uniform(-0.02, 0.02, N_PANELS)
    t = np.arange(N_OBSERVATIONS)
    values = intercepts[:, None] + slopes[:, None] * t
    parameters = pd.DataFrame({"panel": PANEL_IDS, "m": slopes, "b": intercepts})
    return _panel_frame(values, dates), parameters


def generate_sinusoidal(rng: np.random.Generator, dates: pd.DatetimeIndex):
    amplitude = rng.uniform(0.5, 2.0, N_PANELS)
    wavelength = rng.uniform(12.0, 96.0, N_PANELS)
    phase = rng.uniform(0.0, 2.0 * np.pi, N_PANELS)
    t = np.arange(N_OBSERVATIONS)
    values = amplitude[:, None] * np.sin(2.0 * np.pi * t / wavelength[:, None] + phase[:, None])
    parameters = pd.DataFrame(
        {"panel": PANEL_IDS, "amplitude": amplitude, "wavelength": wavelength, "phase": phase}
    )
    return _panel_frame(values, dates), parameters


def generate_ar1(rng: np.random.Generator, dates: pd.DatetimeIndex):
    alpha = rng.uniform(-0.2, 0.2, N_PANELS)
    rho = rng.uniform(0.55, 0.90, N_PANELS)
    sigma = rng.uniform(0.05, 0.20, N_PANELS)
    innovations = rng.normal(0.0, sigma[:, None], size=(N_PANELS, N_OBSERVATIONS))
    values = np.empty_like(innovations)
    previous = alpha / (1.0 - rho)
    for t in range(N_OBSERVATIONS):
        values[:, t] = alpha + rho * previous + innovations[:, t]
        previous = values[:, t]
    parameters = pd.DataFrame({"panel": PANEL_IDS, "alpha": alpha, "rho": rho, "sigma": sigma})
    return _panel_frame(values, dates), parameters


def _write_and_validate(data: pd.DataFrame, parameters: pd.DataFrame, data_path: Path, parameter_path: Path) -> None:
    data.to_csv(data_path, index=False, float_format="%.12f")
    parameters.to_csv(parameter_path, index=False, float_format="%.12f")
    loaded = pd.read_csv(data_path)
    prepared = prepare_panel(loaded, ["panel"], "date", "value", "monthly")
    assert prepared["panel"].nunique() == N_PANELS
    assert prepared.groupby("panel").size().eq(N_OBSERVATIONS).all()
    X, y, _ = make_panel_windows(loaded, ["panel"], "date", "value", "monthly", WINDOW_LENGTH)
    assert X.shape == (N_PANELS * (N_OBSERVATIONS - WINDOW_LENGTH), WINDOW_LENGTH, 1)
    assert y.shape == (N_PANELS * (N_OBSERVATIONS - WINDOW_LENGTH), 1)


def generate_datasets(output_dir: str | Path) -> None:
    """Generate all panel CSVs and validate them through the public dataprep API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START_DATE, periods=N_OBSERVATIONS, freq="MS")
    rng = np.random.default_rng(SEED)
    generators = {
        "linear": generate_linear,
        "sinusoidal": generate_sinusoidal,
        "ar1": generate_ar1,
    }
    for name, generator in generators.items():
        data, parameters = generator(rng, dates)
        _write_and_validate(
            data,
            parameters,
            output_dir / f"{name}_panel.csv",
            output_dir / f"{name}_parameters.csv",
        )


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    generate_datasets(repository_root / "data" / "testdata" / "panels")
