"""Verification harness for make_dataset.py.

Every check re-derives the defect from the DATA and compares it to the MANIFEST.
If the generator and the manifest ever disagree, a test fails.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_gen"))
from make_dataset import NULLABLE, TODAY, build  # noqa: E402


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    df, manifest = build(n_rows=1000, seed=42)
    # Round-trip through CSV: the auditor will read a file, not an in-memory frame.
    path = tmp_path_factory.mktemp("data") / "inv.csv"
    df.to_csv(path, index=False)
    return pd.read_csv(path), manifest


def expected_completeness(manifest: dict, columns: list[str]) -> dict[str, float]:
    """Ground-truth completeness computed ONLY from the manifest — never from the
    data. The auditor's profiler must match this exactly."""
    n = manifest["n_total_rows"]
    return {c: 1 - manifest["nulls"].get(c, {"count": 0})["count"] / n for c in columns}


def test_row_count(generated):
    df, m = generated
    assert len(df) == m["n_total_rows"] == 1020


def test_sku_never_null(generated):
    df, _ = generated
    assert df["sku"].isna().sum() == 0


def test_null_counts_exact(generated):
    df, m = generated
    for col in NULLABLE:
        assert df[col].isna().sum() == m["nulls"].get(col, {"count": 0})["count"], col


def test_expected_completeness_matches_data(generated):
    df, m = generated
    exp = expected_completeness(m, list(df.columns))
    for col, val in exp.items():
        assert df[col].notna().mean() == pytest.approx(val, abs=1e-12), col


def test_duplicates_exact(generated):
    df, m = generated
    assert df.duplicated().sum() == m["duplicates"]["count"]
    # and the duplicate copies are exactly the rows the manifest says
    assert sorted(df.index[df.duplicated()]) == m["duplicates"]["copy_rows"]


def test_negative_quantity(generated):
    df, m = generated
    assert sorted(df.index[df["quantity"] < 0]) == m["invalid"]["negative_quantity"]["rows"]


def test_mrp_below_cost(generated):
    df, m = generated
    assert sorted(df.index[df["mrp"] < df["unit_cost"]]) == m["invalid"]["mrp_below_cost"]["rows"]


def test_expired_stock(generated):
    df, m = generated
    exp = pd.to_datetime(df["expiry_date"], format="%Y-%m-%d", errors="coerce")
    assert sorted(df.index[exp < pd.Timestamp(TODAY)]) == m["invalid"]["expired_stock"]["rows"]


def test_bad_date_format(generated):
    df, m = generated
    s = df["last_received"]
    bad = s.notna() & pd.to_datetime(s, format="%Y-%m-%d", errors="coerce").isna()
    assert sorted(df.index[bad]) == m["invalid"]["bad_date_format"]["rows"]


def test_deterministic():
    a, _ = build(seed=7)
    b, _ = build(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a, _ = build(seed=1)
    b, _ = build(seed=2)
    assert not a.equals(b)
