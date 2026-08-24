import numpy as np
import pandas as pd
import pytest

from ts_trans.dataprep import make_panel_windows, prepare_panel


def monthly_panel():
    rows = []
    for state, values in [("VA", [20, 21, 22, 23, 24]), ("MD", [10, 11, 12, 13, 14])]:
        for month, value in enumerate(values, start=1):
            rows.append({"state": state, "date": f"2020-{month:02d}", "value": value})
    return pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)


def test_monthly_panel_sorting_windows_and_metadata():
    source = monthly_panel()
    prepared = prepare_panel(source, ["state"], "date", "value", "monthly")
    assert prepared["state"].tolist() == ["MD"] * 5 + ["VA"] * 5
    assert prepared["__ts_trans_time_serial"].tolist() == list(range(600, 605)) * 2

    X, y, metadata = make_panel_windows(source, ["state"], "date", "value", "monthly", 3)
    assert X.shape == (4, 3, 1)
    assert y.shape == (4, 1)
    assert metadata[["state", "forecast_time"]].to_dict("records") == [
        {"state": "MD", "forecast_time": "2020-04"},
        {"state": "MD", "forecast_time": "2020-05"},
        {"state": "VA", "forecast_time": "2020-04"},
        {"state": "VA", "forecast_time": "2020-05"},
    ]
    np.testing.assert_array_equal(X[0, :, 0], [10, 11, 12])
    np.testing.assert_array_equal(y[0], [13])
    np.testing.assert_array_equal(X[2, :, 0], [20, 21, 22])


def test_multiple_cross_section_keys_group_together():
    rows = []
    for state, industry, start in [("MD", "31-33", 0), ("MD", "44-45", 100)]:
        for month in range(1, 5):
            rows.append({"state": state, "industry": industry, "date": f"2020-{month:02d}", "value": start + month})
    X, y, metadata = make_panel_windows(
        pd.DataFrame(rows), ["state", "industry"], "date", "value", "monthly", 3
    )
    assert X.shape == (2, 3, 1)
    assert metadata[["state", "industry"]].to_dict("records") == [
        {"state": "MD", "industry": "31-33"},
        {"state": "MD", "industry": "44-45"},
    ]
    np.testing.assert_array_equal(y[:, 0], [4, 104])


def test_quarterly_panel_is_regular():
    source = pd.DataFrame(
        {
            "entity": ["A"] * 4,
            "period": ["2020-01", "2020-04", "2020-07", "2020-10"],
            "value": [1, 2, 3, 4],
        }
    )
    prepared = prepare_panel(source, ["entity"], "period", "value", "quarterly")
    assert prepared["__ts_trans_time_serial"].tolist() == [200, 201, 202, 203]


@pytest.mark.parametrize(
    "bad_frame, message",
    [
        (pd.DataFrame({"state": ["MD", "MD"], "date": ["2020-01", "2020-01"], "value": [1, 2]}), "duplicate"),
        (pd.DataFrame({"state": ["MD", "MD", "MD"], "date": ["2020-01", "2020-02", "2020-04"], "value": [1, 2, 4]}), "missing period"),
        (pd.DataFrame({"state": ["MD"], "date": ["2020-01"], "value": [np.nan]}), "finite"),
        (pd.DataFrame({"state": ["MD"], "date": ["2020-01"], "value": [np.inf]}), "finite"),
    ],
)
def test_invalid_panel_data_is_rejected(bad_frame, message):
    with pytest.raises(ValueError, match=message):
        prepare_panel(bad_frame, ["state"], "date", "value", "monthly")


def test_unsupported_frequency_and_missing_key_are_rejected():
    source = monthly_panel()
    source.loc[0, "state"] = None
    with pytest.raises(ValueError, match="cross-section"):
        prepare_panel(source, ["state"], "date", "value", "monthly")
    with pytest.raises(ValueError, match="frequency"):
        prepare_panel(monthly_panel(), ["state"], "date", "value", "weekly")


def test_input_is_not_mutated_and_window_length_validation_is_per_series():
    source = monthly_panel()
    original = source.copy(deep=True)
    prepare_panel(source, ["state"], "date", "value", "monthly")
    assert source.equals(original)
    with pytest.raises(ValueError, match="more than window_length"):
        make_panel_windows(source, ["state"], "date", "value", "monthly", 5)


def test_empty_panel_is_rejected_before_window_construction():
    empty = pd.DataFrame(columns=["state", "date", "value"])
    with pytest.raises(ValueError, match="at least one row"):
        make_panel_windows(empty, ["state"], "date", "value", "monthly", 3)


@pytest.mark.parametrize(
    "cross_section_cols, time_col, value_col",
    [
        (["state"], "value", "value"),
        (["date"], "date", "value"),
        (["value"], "date", "value"),
        (["state", "state"], "date", "value"),
    ],
)
def test_column_role_collisions_are_rejected(cross_section_cols, time_col, value_col):
    with pytest.raises(ValueError, match="distinct|duplicate"):
        prepare_panel(monthly_panel(), cross_section_cols, time_col, value_col, "monthly")
