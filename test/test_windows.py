from pathlib import Path

import numpy as np
import pytest

from ts_trans.dataprep import load_value_csv, make_windows


LINEAR_CSV = Path(__file__).parents[1] / "data" / "testdata" / "linear_trend.csv"


def test_linear_fixture_windows_have_expected_values_and_shapes():
    values = load_value_csv(LINEAR_CSV)
    inputs, targets = make_windows(values, 5)

    assert values.size == 120
    assert inputs.shape == (115, 5, 1)
    assert targets.shape == (115, 1)
    np.testing.assert_allclose(inputs[0, :, 0], [1.00, 1.05, 1.10, 1.15, 1.20])
    np.testing.assert_allclose(targets[0], [1.25])
    np.testing.assert_allclose(inputs[-1, :, 0], [6.70, 6.75, 6.80, 6.85, 6.90])
    np.testing.assert_allclose(targets[-1], [6.95])


def test_adjacent_windows_shift_by_one_observation():
    values = load_value_csv(LINEAR_CSV)
    inputs, targets = make_windows(values, 5)

    np.testing.assert_array_equal(inputs[1, :-1, 0], inputs[0, 1:, 0])
    np.testing.assert_array_equal(targets[0], inputs[1, -1])


def test_invalid_window_inputs_fail_clearly():
    values = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="positive"):
        make_windows(values, 0)
    with pytest.raises(ValueError, match="more than"):
        make_windows(values, 5)
    with pytest.raises(ValueError, match="one-dimensional"):
        make_windows(values.reshape(1, -1), 2)
    with pytest.raises(ValueError, match="finite"):
        make_windows([1.0, np.nan, 2.0], 1)


def test_window_generation_is_deterministic():
    values = load_value_csv(LINEAR_CSV)
    first = make_windows(values, 5)
    second = make_windows(values, 5)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
