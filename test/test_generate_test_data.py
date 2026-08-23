import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "generate_test_data.py"
SPEC = importlib.util.spec_from_file_location("generate_test_data", SCRIPT_PATH)
generate_test_data_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generate_test_data_module)

SERIES_LENGTHS = generate_test_data_module.SERIES_LENGTHS
generate_series = generate_test_data_module.generate_series
generate_test_data = generate_test_data_module.generate_test_data


DATA_DIR = Path(__file__).parents[1] / "data" / "testdata"


def test_generated_files_have_expected_shape_and_values():
    for name, length in SERIES_LENGTHS.items():
        path = DATA_DIR / f"{name}.csv"
        assert path.exists()
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == length
        assert list(rows[0]) == ["t", "value"]
        assert all(row["t"] != "" and row["value"] != "" for row in rows)

    assert float((DATA_DIR / "linear_trend.csv").read_text().splitlines()[1].split(",")[1]) == 1.0
    assert generate_series("sine_wave", 1) == [0.0]


def test_regeneration_is_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_test_data(first)
    generate_test_data(second)

    for name in SERIES_LENGTHS:
        assert (first / f"{name}.csv").read_bytes() == (second / f"{name}.csv").read_bytes()
