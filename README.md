# DataForge AI — Pharmacy Data Auditor

A data-quality auditor that scores how "AI-ready" a dataset is, built around a
pharmacy inventory domain (drawn from my 4 years running a pharmacy business in India).

**Status:** Module 1 (Data Auditor) in progress · Modules 2–3 scoped

## How it's tested
Instead of hoping the auditor is right, it's checked against **known ground truth**:

1. `data_gen/make_dataset.py` generates a synthetic pharmacy inventory and injects
   an **exact** number of each defect (nulls, duplicates, negative stock,
   MRP below cost, expired batches, malformed dates). It writes a `manifest.json`
   recording which rows got which defect.
2. The auditor reads the CSV with no knowledge of the generator.
3. Tests assert the auditor's findings equal the manifest — exactly, not approximately.

## Progress
| Component | Status |
|---|---|
| Synthetic data generator + verification harness | ✅ 11 tests passing |
| `completeness()` dimension | ✅ 4 tests passing |
| Validity checks (pharmacy rules in YAML) | ⏳ next |
| Weighted AI-Readiness Score | ⏳ |
| Streamlit dashboard (Module 2) | scoped |
| Report generator (Module 3) | scoped |

## Run it
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe data_gen\make_dataset.py --out data
.venv\Scripts\python.exe -m pytest -q
```

## Design decisions
- **SKU is never nulled** — it's the primary key; a missing key makes duplicate detection ambiguous.
- **Duplicates are added before corruption and protected from it** — otherwise a null on one copy silently breaks the duplicate count.
- **Overall completeness = mean of column scores.** Equal to cells-filled ÷ total cells in a rectangular table; they diverge once columns are weighted by business importance (a missing expiry date matters more than a missing supplier).
