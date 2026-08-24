"""Download and audit monthly BLS State and Area CES employment data.

The script deliberately separates raw API responses from the normalized long-form
CSV.  It requests statewide seasonally adjusted total nonfarm employment for the
50 states and the District of Columbia.  Missing observations are preserved as
missing; no imputation is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
START_YEAR = 2005
END_YEAR = date.today().year
MAX_SERIES_PER_REQUEST = 50
MAX_YEARS_PER_REQUEST = 20
MONTH_PERIODS = {f"M{month:02d}" for month in range(1, 13)}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bls"
PROCESSED_PATH = PROJECT_ROOT / "data" / "real" / "ces_state_total_nonfarm.csv"
COVERAGE_PATH = PROJECT_ROOT / "data" / "real" / "ces_state_total_nonfarm_coverage.csv"
MAPPING_PATH = RAW_DIR / "ces_state_series_mapping.csv"


# The official statewide CES pattern is SMS + two-digit state FIPS +
# 000000000000001: total nonfarm, all employees, seasonally adjusted.
_STATE_FIPS_AND_NAMES = [
    ("01", "Alabama"),
    ("02", "Alaska"),
    ("04", "Arizona"),
    ("05", "Arkansas"),
    ("06", "California"),
    ("08", "Colorado"),
    ("09", "Connecticut"),
    ("10", "Delaware"),
    ("11", "District of Columbia"),
    ("12", "Florida"),
    ("13", "Georgia"),
    ("15", "Hawaii"),
    ("16", "Idaho"),
    ("17", "Illinois"),
    ("18", "Indiana"),
    ("19", "Iowa"),
    ("20", "Kansas"),
    ("21", "Kentucky"),
    ("22", "Louisiana"),
    ("23", "Maine"),
    ("24", "Maryland"),
    ("25", "Massachusetts"),
    ("26", "Michigan"),
    ("27", "Minnesota"),
    ("28", "Mississippi"),
    ("29", "Missouri"),
    ("30", "Montana"),
    ("31", "Nebraska"),
    ("32", "Nevada"),
    ("33", "New Hampshire"),
    ("34", "New Jersey"),
    ("35", "New Mexico"),
    ("36", "New York"),
    ("37", "North Carolina"),
    ("38", "North Dakota"),
    ("39", "Ohio"),
    ("40", "Oklahoma"),
    ("41", "Oregon"),
    ("42", "Pennsylvania"),
    ("44", "Rhode Island"),
    ("45", "South Carolina"),
    ("46", "South Dakota"),
    ("47", "Tennessee"),
    ("48", "Texas"),
    ("49", "Utah"),
    ("50", "Vermont"),
    ("51", "Virginia"),
    ("53", "Washington"),
    ("54", "West Virginia"),
    ("55", "Wisconsin"),
    ("56", "Wyoming"),
]


def state_series_id(state_fips: str) -> str:
    """Return the official statewide SA total-nonfarm CES series ID."""
    return f"SMS{state_fips}000000000000001"


def state_mapping() -> pd.DataFrame:
    """Return the explicit 50-state/DC mapping used by this workflow."""
    return pd.DataFrame(
        [
            {
                "state_fips": fips,
                "state_name": name,
                "series_id": state_series_id(fips),
            }
            for fips, name in _STATE_FIPS_AND_NAMES
        ]
    )


def year_chunks(start_year: int, end_year: int, maximum: int = MAX_YEARS_PER_REQUEST):
    """Yield inclusive year ranges no longer than the BLS request limit."""
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    start = start_year
    while start <= end_year:
        end = min(start + maximum - 1, end_year)
        yield start, end
        start = end + 1


def batches(values: list[str], maximum: int = MAX_SERIES_PER_REQUEST):
    """Yield explicit batches of at most ``maximum`` values."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    for start in range(0, len(values), maximum):
        yield values[start : start + maximum]


def load_bls_api_key(env_path: Path = PROJECT_ROOT / ".env") -> str:
    """Load BLS-API-KEY from the environment or a local dotenv file."""
    key = os.environ.get("BLS-API-KEY")
    if key:
        return key
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() != "BLS-API-KEY":
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if value:
                return value
    raise RuntimeError("BLS-API-KEY is absent; set it in the environment or repo-root .env")


def post_bls_json(series_ids: list[str], start_year: int, end_year: int, api_key: str) -> dict[str, Any]:
    """POST one registered request to the official BLS API."""
    body = json.dumps(
        {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
            "registrationkey": api_key,
        }
    ).encode("utf-8")
    request = Request(BLS_API_URL, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_bls_response(response: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize valid monthly records while preserving missing values."""
    if response.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS request failed with status {response.get('status', 'unknown')}")
    rows: list[dict[str, Any]] = []
    counts = {"malformed_values": 0, "missing_values": 0, "invalid_periods": 0, "annual_average_records": 0}
    for series in response.get("Results", {}).get("series", []):
        series_id = series.get("seriesID")
        for observation in series.get("data", []):
            period = observation.get("period")
            if period == "M13":
                counts["annual_average_records"] += 1
                continue
            if period not in MONTH_PERIODS or not str(observation.get("year", "")).isdigit():
                counts["invalid_periods"] += 1
                continue
            value_text = observation.get("value", "")
            if value_text is None or str(value_text).strip() == "":
                value = np.nan
                counts["missing_values"] += 1
            else:
                try:
                    value = float(value_text)
                    if not np.isfinite(value):
                        raise ValueError
                except (TypeError, ValueError):
                    value = np.nan
                    counts["malformed_values"] += 1
            rows.append(
                {
                    "series_id": series_id,
                    "date": f"{int(observation['year']):04d}-{int(period[1:]):02d}",
                    "value": value,
                    "period": period,
                    "period_name": observation.get("periodName"),
                    "footnotes": observation.get("footnotes", []),
                    "latest": observation.get("latest"),
                }
            )
    columns = ["series_id", "date", "value", "period", "period_name", "footnotes", "latest"]
    return pd.DataFrame(rows, columns=columns), counts


def deduplicate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate agreeing overlaps and reject conflicting observations."""
    if observations.empty:
        return observations.copy()
    key = ["series_id", "date"]
    for _, group in observations.groupby(key, sort=False, dropna=False):
        values = group["value"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) not in {0, len(values)} or (
            len(finite) > 1 and not np.allclose(finite, finite[0], rtol=0, atol=1e-12)
        ):
            series_id, observation_date = group.iloc[0][key].tolist()
            raise ValueError(f"conflicting duplicate observation for {series_id} on {observation_date}")
    return observations.drop_duplicates(key, keep="first").reset_index(drop=True)


def coverage_audit(mapping: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize returned coverage and internal monthly gaps per requested series."""
    rows = []
    for item in mapping.itertuples(index=False):
        series = observations[observations["series_id"] == item.series_id].copy()
        dates = pd.to_datetime(series["date"], format="%Y-%m", errors="coerce").dropna().sort_values().unique()
        missing_internal = 0
        first_date = last_date = ""
        if len(dates):
            first_date = pd.Timestamp(dates[0]).strftime("%Y-%m")
            last_date = pd.Timestamp(dates[-1]).strftime("%Y-%m")
            expected = pd.date_range(dates[0], dates[-1], freq="MS")
            missing_internal = len(expected.difference(pd.DatetimeIndex(dates)))
        missing_values = int(series["value"].isna().sum())
        if series.empty:
            status = "no_data_returned"
        elif missing_internal:
            status = "has_gaps"
        elif missing_values:
            status = "parse_issue"
        else:
            status = "complete"
        rows.append(
            {
                "state_fips": item.state_fips,
                "state_name": item.state_name,
                "series_id": item.series_id,
                "first_date": first_date,
                "last_date": last_date,
                "n_observations": len(series),
                "n_missing_internal_periods": missing_internal,
                "n_missing_values": missing_values,
                "status": status,
                "usable_by_prepare_panel": status == "complete",
            }
        )
    return pd.DataFrame(rows)


def _footnotes(value: Any) -> str:
    if not isinstance(value, list):
        return "" if value is None else str(value)
    return "; ".join(str(item.get("text", "")) for item in value if isinstance(item, dict))


def download(
    api_key: str,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    request_json: Callable[[list[str], int, int, str], dict[str, Any]] = post_bls_json,
) -> dict[str, Any]:
    """Download, normalize, audit, and write the state CES dataset."""
    mapping = state_mapping()
    series_ids = mapping["series_id"].tolist()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_PATH.parent).mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_PATH, index=False)

    response_count = 0
    parsed_parts: list[pd.DataFrame] = []
    parse_counts = {"malformed_values": 0, "missing_values": 0, "invalid_periods": 0, "annual_average_records": 0}
    for year_start, year_end in year_chunks(start_year, end_year):
        for batch_number, batch in enumerate(batches(series_ids), start=1):
            response = request_json(batch, year_start, year_end, api_key)
            response_count += 1
            raw_path = RAW_DIR / f"ces_state_{year_start}_{year_end}_batch{batch_number}.json"
            raw_path.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
            part, counts = parse_bls_response(response)
            parsed_parts.append(part)
            for key, value in counts.items():
                parse_counts[key] += value

    columns = ["series_id", "date", "value", "period", "period_name", "footnotes", "latest"]
    observations = pd.concat(parsed_parts, ignore_index=True) if parsed_parts else pd.DataFrame(columns=columns)
    observations = deduplicate_observations(observations)
    observations["footnotes"] = observations["footnotes"].map(_footnotes)
    coverage = coverage_audit(mapping, observations)
    merged = observations.merge(mapping, on="series_id", how="left", validate="many_to_one")
    output_columns = ["state_fips", "state_name", "series_id", "date", "value", "period", "period_name", "footnotes", "latest"]
    merged = merged[output_columns].sort_values(["state_fips", "date"], kind="mergesort")
    merged.to_csv(PROCESSED_PATH, index=False, na_rep="NaN")
    coverage.to_csv(COVERAGE_PATH, index=False)

    dated = pd.to_datetime(observations["date"], format="%Y-%m", errors="coerce").dropna()
    no_data = int((coverage["status"] == "no_data_returned").sum())
    summary = {
        "start_year": start_year,
        "end_year": end_year,
        "requested_series": len(mapping),
        "api_calls": response_count,
        "series_with_data": int((coverage["n_observations"] > 0).sum()),
        "series_without_data": no_data,
        "total_observations": len(observations),
        "earliest_date": dated.min().strftime("%Y-%m") if len(dated) else "",
        "latest_date": dated.max().strftime("%Y-%m") if len(dated) else "",
        "series_with_gaps": int((coverage["n_missing_internal_periods"] > 0).sum()),
        "usable_by_prepare_panel": int(coverage["usable_by_prepare_panel"].sum()),
        **parse_counts,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    args = parser.parse_args()
    try:
        summary = download(load_bls_api_key(), args.start_year, args.end_year)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Processed data: {PROCESSED_PATH}")
    print(f"Coverage audit: {COVERAGE_PATH}")
    print(f"Raw responses: {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
