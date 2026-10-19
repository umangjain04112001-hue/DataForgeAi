"""
dimensions.py — Each function scores one data-quality dimension.

Rules for every dimension function:
  - input: a pandas DataFrame (as read from CSV)
  - output: a dict with a 0–1 score AND the evidence behind it
  - no hardcoded knowledge of make_dataset.py — the auditor must work on any file
"""
import pandas as pd


def completeness(df: pd.DataFrame) -> dict:
    """
    Share of non-null cells, per column and overall.

    Overall score = mean of column scores. In a rectangular table this equals
    non-null cells / total cells, because every column has the same row count.
    They diverge once columns are weighted by importance (e.g. a missing
    expiry_date matters more than a missing supplier) - handled in score.py.
    """
    per_column = df.notna().mean().to_dict()
    score = sum(per_column.values()) / len(per_column)
    worst_column = min(per_column, key=per_column.get)
    return {"score": score, "per_column": per_column, "worst_column": worst_column}
