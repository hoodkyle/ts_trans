import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_ces_state_holdout.py"
SPEC = importlib.util.spec_from_file_location("train_ces_state_holdout", SCRIPT_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def sample_data():
    dates = pd.date_range("2020-01", periods=8, freq="MS")
    rows = []
    for state, start in [("01", 100.0), ("02", 200.0)]:
        values = start + np.arange(len(dates))
        rows.extend(
            {"state_fips": state, "state_name": state, "series_id": f"S{state}", "date": date.strftime("%Y-%m"), "value": value}
            for date, value in zip(dates, values)
        )
    return pd.DataFrame(rows)


def test_log_difference_and_split_are_chronological():
    transformed = experiment.make_log_difference(sample_data())
    assert transformed["log_difference"].isna().sum() == 2
    expected = np.log(101.0) - np.log(100.0)
    assert transformed.loc[1, "log_difference"] == pytest.approx(expected)
    train, holdout = experiment.chronological_split(transformed, "2020-04", "2020-05")
    assert train["date"].max() == pd.Timestamp("2020-04-01")
    assert holdout["date"].min() == pd.Timestamp("2020-05-01")


def test_scaling_is_fit_on_training_rows_only_and_holdout_changes_do_not_leak():
    transformed = experiment.make_log_difference(sample_data())
    train, holdout = experiment.chronological_split(transformed, "2020-04", "2020-05")
    first_scaling = experiment.fit_state_scaling(train)
    changed_holdout = holdout.copy()
    changed_holdout["log_difference"] = changed_holdout["log_difference"] * 1000
    changed_train, changed_holdout = experiment.chronological_split(
        pd.concat([train, changed_holdout], ignore_index=True), "2020-04", "2020-05"
    )
    second_scaling = experiment.fit_state_scaling(changed_train)
    pd.testing.assert_frame_equal(first_scaling, second_scaling)
    first_ar1 = experiment.fit_ar1(train)
    second_ar1 = experiment.fit_ar1(changed_train)
    pd.testing.assert_frame_equal(first_ar1, second_ar1)
    applied = experiment.apply_state_scaling(pd.concat([train, changed_holdout], ignore_index=True), first_scaling)
    assert np.isfinite(applied["scaled_log_difference"].dropna()).all()


def test_ar1_fit_and_level_inversion_are_explicit():
    dates = pd.date_range("2020-01", periods=8, freq="MS")
    growth = np.array([np.nan, 1.0, 1.5, 1.75, 1.875, 1.9375, 1.96875, 1.984375])
    data = pd.DataFrame(
        {
            "state_fips": "01",
            "date": dates,
            "log_difference": growth,
        }
    )
    coefficients = experiment.fit_ar1(data)
    assert coefficients.loc[0, "alpha"] == pytest.approx(1.0)
    assert coefficients.loc[0, "rho"] == pytest.approx(0.5)
    data["state_name"] = "state"
    forecasts = experiment.forecast_ar1(data, coefficients, "2020-05")
    np.testing.assert_allclose(forecasts, [1.875, 1.9375, 1.96875, 1.984375])
    restored = experiment.invert_log_difference(np.array([0.0, np.log(2.0)]), np.array([10.0, 10.0]))
    np.testing.assert_allclose(restored, [10.0, 20.0])


def test_nonpositive_levels_are_rejected_before_log():
    bad = sample_data()
    bad.loc[0, "value"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        experiment.make_log_difference(bad)
