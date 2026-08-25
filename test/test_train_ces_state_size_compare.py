import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_ces_state_size_compare.py"
SPEC = importlib.util.spec_from_file_location("train_ces_state_size_compare", SCRIPT_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(experiment)


def test_model_width_sets_feed_forward_width():
    assert experiment.model_config(4) == (4, 8)
    assert experiment.model_config(8) == (8, 16)
    assert experiment.model_config(16) == (16, 32)


def test_model_width_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        experiment.model_config(0)
