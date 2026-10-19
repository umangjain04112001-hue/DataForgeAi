"""
make_dataset.py — synthetic pharmacy inventory generator with EXACT-COUNT corruption.

Why exact counts: the auditor's scores must be testable with `==`, not "roughly".
Every defect is injected a known number of times and recorded in a manifest
(ground truth). Tests compare the auditor's findings against the manifest.

Pipeline order (matters — see DESIGN NOTES at the bottom):
    1. generate clean rows
    2. append exact duplicates
    3. inject nulls          (never into `sku`)
    4. inject invalid values (negative qty, mrp < cost, expired stock, bad date strings)

Run:
    python data_gen/make_dataset.py --rows 1000 --seed 42 --out data/
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- reference data ---------------------------------------------------------
DRUGS = [
    ("Paracetamol 500mg", "Analgesic"), ("Ibuprofen 400mg", "Analgesic"),
    ("Amoxicillin 500mg", "Antibiotic"), ("Azithromycin 250mg", "Antibiotic"),
    ("Metformin 500mg", "Antidiabetic"), ("Glimepiride 2mg", "Antidiabetic"),
    ("Amlodipine 5mg", "Antihypertensive"), ("Telmisartan 40mg", "Antihypertensive"),
    ("Atorvastatin 10mg", "Statin"), ("Pantoprazole 40mg", "Antacid"),
    ("Cetirizine 10mg", "Antihistamine"), ("Montelukast 10mg", "Antiasthmatic"),
]
SUPPLIERS = ["Sun Distributors", "MedPlus Wholesale", "Apollo Supply", "Cipla Direct", "Local Stockist"]

COLUMNS = ["sku", "drug_name", "category", "batch_no", "expiry_date",
           "quantity", "unit_cost", "mrp", "supplier", "last_received"]

# Columns eligible for null injection. `sku` is deliberately excluded (see notes).
NULLABLE = ["drug_name", "category", "batch_no", "expiry_date",
            "quantity", "unit_cost", "mrp", "supplier", "last_received"]

TODAY = date(2026, 10, 1)  # fixed "as-of" date so expiry checks are reproducible


# --- step 1: clean data -----------------------------------------------------
def make_clean(n: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.integers(0, len(DRUGS), n)
    unit_cost = rng.uniform(1.0, 50.0, n).round(2)
    markup = rng.uniform(1.15, 1.60, n)  # MRP always above cost in clean data
    received = [TODAY - timedelta(days=int(d)) for d in rng.integers(1, 180, n)]
    expiry = [TODAY + timedelta(days=int(d)) for d in rng.integers(60, 900, n)]

    return pd.DataFrame({
        "sku": [f"SKU{i:05d}" for i in range(n)],
        "drug_name": [DRUGS[i][0] for i in idx],
        "category": [DRUGS[i][1] for i in idx],
        "batch_no": [f"B{rng.integers(10000, 99999)}" for _ in range(n)],
        "expiry_date": [d.isoformat() for d in expiry],
        "quantity": rng.integers(0, 500, n).astype("float"),
        "unit_cost": unit_cost,
        "mrp": (unit_cost * markup).round(2),
        "supplier": rng.choice(SUPPLIERS, n),
        "last_received": [d.isoformat() for d in received],
    }, columns=COLUMNS)


# --- step 2: duplicates -----------------------------------------------------
def add_duplicates(df: pd.DataFrame, k: int, rng: np.random.Generator) -> tuple[pd.DataFrame, list[int]]:
    """Append k exact copies of k distinct clean rows."""
    src = rng.choice(len(df), size=k, replace=False)
    dupes = df.iloc[src].copy()
    out = pd.concat([df, dupes], ignore_index=True)
    return out, sorted(int(i) for i in src)


# --- step 3: nulls ----------------------------------------------------------
def inject_nulls(df: pd.DataFrame, per_col: dict[str, int], rng: np.random.Generator,
                 protected: set[int]) -> dict[str, list[int]]:
    """Set exactly per_col[c] cells to NaN in column c. Rows in `protected`
    (duplicate pairs) are skipped so duplicates stay exact duplicates."""
    hits: dict[str, list[int]] = {}
    pool = np.array([i for i in range(len(df)) if i not in protected])
    for col, k in per_col.items():
        assert col in NULLABLE, f"{col} is not nullable"
        rows = rng.choice(pool, size=k, replace=False)
        df.loc[rows, col] = np.nan
        hits[col] = sorted(int(r) for r in rows)
    return hits


# --- step 4: invalid values -------------------------------------------------
def inject_invalid(df: pd.DataFrame, counts: dict[str, int], rng: np.random.Generator,
                   protected: set[int], null_rows: dict[str, list[int]]) -> dict[str, list[int]]:
    """Each defect type hits rows that are (a) not duplicates, (b) not already
    null in the affected column(s), (c) not already hit by another defect.
    That keeps every defect independently countable."""
    used: set[int] = set()
    hits: dict[str, list[int]] = {}

    def pick(k: int, cols: list[str]) -> np.ndarray:
        blocked = protected | used
        for c in cols:
            blocked |= set(null_rows.get(c, []))
        pool = np.array([i for i in range(len(df)) if i not in blocked])
        rows = rng.choice(pool, size=k, replace=False)
        used.update(int(r) for r in rows)
        return rows

    rows = pick(counts["negative_quantity"], ["quantity"])
    df.loc[rows, "quantity"] = -rng.integers(1, 50, len(rows)).astype(float)
    hits["negative_quantity"] = sorted(map(int, rows))

    rows = pick(counts["mrp_below_cost"], ["mrp", "unit_cost"])
    df.loc[rows, "mrp"] = (df.loc[rows, "unit_cost"] * 0.8).round(2)
    hits["mrp_below_cost"] = sorted(map(int, rows))

    rows = pick(counts["expired_stock"], ["expiry_date"])
    df.loc[rows, "expiry_date"] = [(TODAY - timedelta(days=int(d))).isoformat()
                                   for d in rng.integers(1, 120, len(rows))]
    hits["expired_stock"] = sorted(map(int, rows))

    rows = pick(counts["bad_date_format"], ["last_received"])
    df.loc[rows, "last_received"] = [pd.Timestamp(d).strftime("%d/%m/%Y")
                                     for d in df.loc[rows, "last_received"]]
    hits["bad_date_format"] = sorted(map(int, rows))
    return hits


# --- orchestration ----------------------------------------------------------
DEFAULT_SPEC = {
    "duplicates": 20,
    "nulls": {"drug_name": 15, "batch_no": 30, "expiry_date": 25, "quantity": 10,
              "unit_cost": 12, "mrp": 8, "supplier": 40, "last_received": 18},
    "invalid": {"negative_quantity": 7, "mrp_below_cost": 9,
                "expired_stock": 14, "bad_date_format": 11},
}


def build(n_rows: int = 1000, seed: int = 42, spec: dict | None = None) -> tuple[pd.DataFrame, dict]:
    spec = spec or DEFAULT_SPEC
    rng = np.random.default_rng(seed)

    df = make_clean(n_rows, rng)
    df, dup_src = add_duplicates(df, spec["duplicates"], rng)
    dup_copies = list(range(n_rows, n_rows + spec["duplicates"]))
    protected = set(dup_src) | set(dup_copies)

    null_hits = inject_nulls(df, spec["nulls"], rng, protected)
    invalid_hits = inject_invalid(df, spec["invalid"], rng, protected, null_hits)

    manifest = {
        "seed": seed,
        "as_of": TODAY.isoformat(),
        "n_clean_rows": n_rows,
        "n_total_rows": len(df),
        "duplicates": {"count": spec["duplicates"], "source_rows": dup_src, "copy_rows": dup_copies},
        "nulls": {c: {"count": len(r), "rows": r} for c, r in null_hits.items()},
        "invalid": {k: {"count": len(r), "rows": r} for k, r in invalid_hits.items()},
    }
    return df, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()

    df, manifest = build(a.rows, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "pharmacy_inventory.csv", index=False)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(df)} rows -> {a.out/'pharmacy_inventory.csv'}")


if __name__ == "__main__":
    main()

# DESIGN NOTES (you should be able to say these out loud in an interview)
# - SKU is never nulled: it's the primary key. If the key itself is missing you
#   can't tell a duplicate from a different item, so duplicate counts become
#   ambiguous. Key integrity is a separate check from completeness.
# - Duplicates are appended BEFORE corruption and protected from it: otherwise
#   a null landing on one copy makes the pair no longer an exact duplicate, and
#   the expected duplicate count silently drops.
# - Invalid-value defects never overlap each other or a null in the same column,
#   so each defect is counted exactly once.
