import numpy as np
import pandas as pd
import pytest

from ts_trans.dataprep import (
    inverse_standardize,
    make_panel_windows,
    standardize_panel,
)


def simple_panel():
    return pd.DataFrame(
        {
            "panel": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )


def test_standardize_panel_uses_per_panel_population_statistics():
    source = simple_panel()
    original = source.copy(deep=True)
    scaled, parameters = standardize_panel(source, ["panel"], "value")

    assert source.equals(original)
    assert parameters["panel"].tolist() == ["A", "B"]
    assert parameters.loc[0, "mean"] == 2.0
    assert parameters.loc[0, "std"] == pytest.approx(np.sqrt(2 / 3))
    for _, group in scaled.groupby("panel"):
        assert group["value"].mean() == pytest.approx(0.0)
        assert group["value"].std(ddof=0) == pytest.approx(1.0)


def test_standardize_panel_uses_full_multiple_key_combination():
    source = pd.DataFrame(
        {
            "state": ["MD", "MD", "MD", "MD"],
            "industry": ["31", "31", "44", "44"],
            "value": [1.0, 3.0, 10.0, 30.0],
        }
    )
    scaled, parameters = standardize_panel(source, ["state", "industry"], "value")

    assert len(parameters) == 2
    assert scaled.groupby(["state", "industry"])["value"].mean().abs().max() == pytest.approx(0.0)
    assert scaled.groupby(["state", "industry"])["value"].std(ddof=0).min() == pytest.approx(1.0)


def test_inverse_standardize_round_trip_and_shape_preservation():
    source = simple_panel()
    scaled, parameters = standardize_panel(source, ["panel"], "value")
    metadata = source[["panel"]].copy()
    standardized = scaled["value"].to_numpy()

    restored_1d = inverse_standardize(standardized, metadata, parameters, ["panel"])
    restored_2d = inverse_standardize(standardized[:, None], metadata, parameters, ["panel"])
    np.testing.assert_allclose(restored_1d, source["value"].to_numpy())
    np.testing.assert_allclose(restored_2d[:, 0], source["value"].to_numpy())
    assert restored_1d.shape == (6,)
    assert restored_2d.shape == (6, 1)


def test_window_targets_can_be_inverted_to_original_units():
    raw = pd.DataFrame(
        {
            "panel": ["A"] * 5 + ["B"] * 5,
            "date": [f"2020-{month:02d}" for month in range(1, 6)] * 2,
            "value": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        }
    )
    scaled, parameters = standardize_panel(raw, ["panel"], "value")
    X, y_scaled, metadata = make_panel_windows(scaled, ["panel"], "date", "value", "monthly", 3)
    _, y_original, _ = make_panel_windows(raw, ["panel"], "date", "value", "monthly", 3)
    restored = inverse_standardize(y_scaled, metadata, parameters, ["panel"])

    assert X.shape == (4, 3, 1)
    np.testing.assert_allclose(restored, y_original)


def test_constant_panel_is_rejected():
    with pytest.raises(ValueError, match="zero or near-zero"):
        standardize_panel(pd.DataFrame({"panel": ["A", "A"], "value": [2.0, 2.0]}), ["panel"], "value")


def test_inverse_rejects_bad_metadata():
    source = simple_panel()
    _, parameters = standardize_panel(source, ["panel"], "value")
    values = np.zeros(2)
    with pytest.raises(ValueError, match="metadata is missing"):
        inverse_standardize(values, pd.DataFrame({"other": ["A", "B"]}), parameters, ["panel"])
    with pytest.raises(ValueError, match="no scaling parameters"):
        inverse_standardize(values, pd.DataFrame({"panel": ["A", "C"]}), parameters, ["panel"])
    with pytest.raises(ValueError, match="must match"):
        inverse_standardize(values, pd.DataFrame({"panel": ["A"]}), parameters, ["panel"])


def test_invalid_scaling_inputs_are_rejected():
    with pytest.raises(ValueError, match="cross-section"):
        standardize_panel(pd.DataFrame({"panel": [None], "value": [1.0]}), ["panel"], "value")
    with pytest.raises(ValueError, match="finite"):
        standardize_panel(pd.DataFrame({"panel": ["A"], "value": [np.inf]}), ["panel"], "value")
