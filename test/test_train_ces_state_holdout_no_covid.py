import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_ces_state_holdout_no_covid.py"
SPEC = importlib.util.spec_from_file_location("train_ces_state_holdout_no_covid", SCRIPT_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def sample_data():
    dates = pd.date_range("2019-01", periods=48, freq="MS")
    return pd.DataFrame(
        {
            "state_fips": ["01"] * len(dates),
            "date": dates,
            "log_difference": 0.001,
            "state_name": "Alabama",
        }
    )


def test_excluded_target_dates_are_not_eligible():
    eligible = experiment.eligible_training_rows(sample_data())
    assert not experiment.excluded_target_mask(eligible["date"]).any()
    assert eligible["date"].max() == pd.Timestamp("2022-12-01")


def test_window_filter_removes_only_excluded_targets_and_preserves_holdout():
    metadata = pd.DataFrame(
        {"forecast_date": pd.to_datetime(["2020-02", "2020-03", "2021-12", "2022-01", "2023-01"])}
    )
    X = pd.DataFrame({"x": range(5)}).to_numpy()[:, None, :]
    y = pd.DataFrame({"y": range(5)}).to_numpy()
    kept_X, kept_y, kept_metadata = experiment.filter_training_windows(X, y, metadata)
    assert kept_metadata["forecast_date"].dt.strftime("%Y-%m").tolist() == ["2020-02", "2022-01", "2023-01"]
    assert len(kept_X) == len(kept_y) == 3


def test_exclusion_bounds_are_inclusive():
    dates = pd.to_datetime(["2020-02", "2020-03", "2021-12", "2022-01"])
    assert experiment.excluded_target_mask(dates).tolist() == [False, True, True, False]
