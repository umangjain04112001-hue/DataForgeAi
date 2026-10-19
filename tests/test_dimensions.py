"""Tests for YOUR auditor code. These fail until you implement dimensions.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_gen"))
sys.path.insert(0, str(ROOT))
from make_dataset import build  # noqa: E402
from auditor.dimensions import completeness  # noqa: E402
from tests.test_make_dataset import expected_completeness  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    df, m = build(n_rows=1000, seed=42)
    p = tmp_path_factory.mktemp("d") / "inv.csv"
    df.to_csv(p, index=False)
    return pd.read_csv(p), m


def test_completeness_per_column_matches_manifest(data):
    df, m = data
    got = completeness(df)["per_column"]
    exp = expected_completeness(m, list(df.columns))
    for col in df.columns:
        assert got[col] == pytest.approx(exp[col], abs=1e-12), col


def test_completeness_worst_column(data):
    df, _ = data
    assert completeness(df)["worst_column"] == "supplier"  # 40 nulls, the most


def test_completeness_score_in_range(data):
    df, _ = data
    s = completeness(df)["score"]
    assert 0.0 <= s <= 1.0


def test_completeness_clean_frame_is_perfect():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert completeness(df)["score"] == 1.0
