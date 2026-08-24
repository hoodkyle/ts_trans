import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_ces_state.py"
SPEC = importlib.util.spec_from_file_location("download_ces_state", SCRIPT_PATH)
ces = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ces)


def response_for(data):
    return {"status": "REQUEST_SUCCEEDED", "Results": {"series": data}}


def test_months_are_canonical_and_m13_is_excluded():
    data = [{"seriesID": "SMS06000000000000001", "data": [
        {"year": "2024", "period": f"M{month:02d}", "periodName": "month", "value": str(month)}
        for month in range(1, 13)
    ] + [{"year": "2024", "period": "M13", "periodName": "Annual", "value": "99"}]}]
    records, counts = ces.parse_bls_response(response_for(data))
    assert records["date"].tolist() == [f"2024-{month:02d}" for month in range(1, 13)]
    assert counts["annual_average_records"] == 1


def test_blank_and_malformed_values_are_missing_not_zero():
    data = [{"seriesID": "SMS01000000000000001", "data": [
        {"year": "2024", "period": "M01", "value": ""},
        {"year": "2024", "period": "M02", "value": "not-a-number"},
        {"year": "2024", "period": "M03", "value": "12.5"},
    ]}]
    records, counts = ces.parse_bls_response(response_for(data))
    assert records["value"].isna().tolist() == [True, True, False]
    assert not (records["value"].fillna(-1) == 0).any()
    assert counts["missing_values"] == 1
    assert counts["malformed_values"] == 1


def test_missing_period_is_reported_without_inserting_a_zero():
    data = [{"seriesID": "SMS01000000000000001", "data": [
        {"year": "2024", "period": "M01", "value": "1"},
        {"year": "2024", "period": "M02", "value": "2"},
        {"year": "2024", "period": "M04", "value": "4"},
    ]}]
    records, _ = ces.parse_bls_response(response_for(data))
    coverage = ces.coverage_audit(
        pd.DataFrame([{"state_fips": "01", "state_name": "Alabama", "series_id": "SMS01000000000000001"}]),
        records,
    )
    assert records["date"].tolist() == ["2024-01", "2024-02", "2024-04"]
    assert coverage.loc[0, "n_missing_internal_periods"] == 1
    assert coverage.loc[0, "status"] == "has_gaps"


def test_missing_series_appears_in_coverage_but_not_observations():
    mapping = pd.DataFrame([{"state_fips": "01", "state_name": "Alabama", "series_id": "SMS01000000000000001"}])
    records, _ = ces.parse_bls_response(response_for([]))
    coverage = ces.coverage_audit(mapping, records)
    assert records.empty
    assert coverage.loc[0, "status"] == "no_data_returned"


def test_agreeing_duplicates_deduplicate_and_conflicts_fail():
    records = pd.DataFrame([
        {"series_id": "S", "date": "2024-01", "value": 1.0},
        {"series_id": "S", "date": "2024-01", "value": 1.0},
    ])
    assert len(ces.deduplicate_observations(records)) == 1
    conflicting = records.copy()
    conflicting.loc[1, "value"] = 2.0
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ces.deduplicate_observations(conflicting)
    missing_conflict = records.copy()
    missing_conflict.loc[1, "value"] = np.nan
    with pytest.raises(ValueError, match="conflicting duplicate"):
        ces.deduplicate_observations(missing_conflict)


def test_mapping_and_request_limits_are_explicit():
    mapping = ces.state_mapping()
    assert len(mapping) == 51
    assert mapping["series_id"].is_unique
    assert list(ces.year_chunks(2005, 2026, 20)) == [(2005, 2024), (2025, 2026)]
    assert [len(batch) for batch in ces.batches(mapping["series_id"].tolist())] == [50, 1]
