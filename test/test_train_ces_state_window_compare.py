import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_ces_state_window_compare.py"
SPEC = importlib.util.spec_from_file_location("train_ces_state_window_compare", SCRIPT_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def sample_data():
    dates = pd.date_range("2019-01", periods=30, freq="MS")
    return pd.DataFrame(
        {
            "state_fips": "01",
            "state_name": "Alabama",
            "date": dates,
            "level": np.arange(100.0, 130.0),
            "log_difference": np.r_[np.nan, np.repeat(0.01, 29)],
            "scaled_log_difference": np.r_[np.nan, np.repeat(0.01, 29)],
        }
    )


def test_window_lengths_change_training_windows_but_not_target_exclusion():
    data = sample_data()
    X6, y6, m6 = experiment.build_windows_for_length(data, 6, "2020-01", "2021-06")
    X12, y12, m12 = experiment.build_windows_for_length(data, 12, "2020-01", "2021-06")
    assert X6.shape[1] == 6
    assert X12.shape[1] == 12
    assert len(y6) > len(y12)
    filtered_X, filtered_y, filtered_m = experiment.filter_excluded_targets(X6, y6, m6)
    assert len(filtered_X) == len(filtered_y) == len(filtered_m)
    assert not filtered_m["forecast_date"].between("2020-03", "2021-12").any()


def test_invalid_window_length_fails_clearly():
    with pytest.raises(ValueError, match="positive"):
        experiment.build_windows_for_length(sample_data(), 0, "2020-01")
