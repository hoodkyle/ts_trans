import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from ts_trans.dataprep import make_panel_windows, prepare_panel


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "generate_panel_test_data.py"
SPEC = importlib.util.spec_from_file_location("generate_panel_test_data", SCRIPT_PATH)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)

N_OBSERVATIONS = generator.N_OBSERVATIONS
N_PANELS = generator.N_PANELS
SEED = generator.SEED
generate_ar1 = generator.generate_ar1
generate_datasets = generator.generate_datasets


DATA_DIR = Path(__file__).parents[1] / "data" / "testdata" / "panels"
DATA_FILES = ["linear_panel.csv", "sinusoidal_panel.csv", "ar1_panel.csv"]
PARAMETER_FILES = ["linear_parameters.csv", "sinusoidal_parameters.csv", "ar1_parameters.csv"]


def test_generated_panel_files_have_expected_sizes_and_pass_dataprep():
    for filename in DATA_FILES:
        data = pd.read_csv(DATA_DIR / filename)
        assert list(data.columns) == ["panel", "date", "value"]
        assert len(data) == N_PANELS * N_OBSERVATIONS
        assert data["panel"].nunique() == N_PANELS
        assert data.groupby("panel").size().eq(N_OBSERVATIONS).all()
        prepared = prepare_panel(data, ["panel"], "date", "value", "monthly")
        X, y, _ = make_panel_windows(data, ["panel"], "date", "value", "monthly", 12)
        assert len(prepared) == len(data)
        assert X.shape == (22800, 12, 1)
        assert y.shape == (22800, 1)

    for filename in PARAMETER_FILES:
        parameters = pd.read_csv(DATA_DIR / filename)
        assert len(parameters) == N_PANELS
        assert parameters["panel"].tolist() == [f"P{i:03d}" for i in range(N_PANELS)]


def test_generated_files_are_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_datasets(first)
    generate_datasets(second)
    for filename in DATA_FILES + PARAMETER_FILES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_linear_values_match_saved_parameters():
    data = pd.read_csv(DATA_DIR / "linear_panel.csv")
    parameters = pd.read_csv(DATA_DIR / "linear_parameters.csv").set_index("panel")
    for panel in ["P000", "P050", "P099"]:
        values = data[data["panel"] == panel]["value"].to_numpy()
        t = np.arange(N_OBSERVATIONS)
        expected = parameters.loc[panel, "b"] + parameters.loc[panel, "m"] * t
        np.testing.assert_allclose(values, expected, atol=1e-10)


def test_sinusoidal_values_match_saved_parameters():
    data = pd.read_csv(DATA_DIR / "sinusoidal_panel.csv")
    parameters = pd.read_csv(DATA_DIR / "sinusoidal_parameters.csv").set_index("panel")
    for panel in ["P000", "P050", "P099"]:
        values = data[data["panel"] == panel]["value"].to_numpy()
        row = parameters.loc[panel]
        t = np.arange(N_OBSERVATIONS)
        expected = row["amplitude"] * np.sin(2 * np.pi * t / row["wavelength"] + row["phase"])
        np.testing.assert_allclose(values, expected, atol=1e-10)


def test_ar1_regeneration_is_reproducible_from_fixed_seed():
    dates = pd.date_range("2000-01", periods=N_OBSERVATIONS, freq="MS")
    first, first_parameters = generate_ar1(np.random.default_rng(SEED), dates)
    second, second_parameters = generate_ar1(np.random.default_rng(SEED), dates)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_parameters, second_parameters)
