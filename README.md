# Drug-Associated Kidney Stone Risk in NHANES

Analysis pipeline for estimating associations between prescription drug classes and self-reported kidney stone history in NHANES 2007-2018.

The main analysis uses a target-trial-style framing with augmented inverse probability weighting (AIPW), covariate balance diagnostics, E-values, and duration-based exposure sensitivity checks.

## What This Produces

- `data/processed/analysis_ready.parquet`: merged, cleaned, imputed analysis dataset
- `outputs/tables/executive_summary.csv`: primary effect estimates and diagnostics
- `outputs/tables/ate_summary.csv`: AIPW estimates, confidence intervals, FDR correction, and E-values
- `outputs/tables/covariate_balance.csv`: pre/post-weighting standardized mean differences
- `outputs/tables/ate_duration_sensitivity.csv`: `RXDDAYS >= 365` and `RXDDAYS >= 730` sensitivity analyses
- `outputs/figures/main_figure_panel.png`: compact results panel

Raw NHANES files and generated outputs are intentionally ignored by git. Recreate them with the commands below.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost may also require:

```bash
brew install libomp
```

## Run

If `data/raw` already contains the NHANES `.xpt` files, start at preprocessing:

```bash
python 02_preprocess.py
python 03_eda.py
python 04_causal_analysis.py
python 06_results.py
```

To refresh raw inputs first:

```bash
python 01_download_data.py
```

`05_prediction.py` is optional. It generates supplementary XGBoost/logistic prediction artifacts and SHAP plots, but prediction is not the primary analysis.

## Estimand

- Population: NHANES adults age 20+ from 2007-2018 with valid kidney stone questionnaire responses
- Time zero: NHANES household interview and prescription medication inventory date
- Exposure: any reported current use of each prescription drug class in the prior 30 days
- Outcome: lifetime self-reported kidney stone history at time zero (`KIQ026`)
- Sensitivity exposure windows: current use with `RXDDAYS >= 365` or `RXDDAYS >= 730`

Important limitation: NHANES is cross-sectional. These estimates should be read as adjusted prevalence effects/associations under causal assumptions, not as definitive prospective drug effects on incident stones.

## Pipeline

| Script | Purpose |
|---|---|
| `01_download_data.py` | Create data/output directories and fetch or copy NHANES XPT files |
| `02_preprocess.py` | Merge NHANES components, engineer features, impute missing values |
| `03_eda.py` | Generate descriptive tables and exploratory figures |
| `04_causal_analysis.py` | Estimate drug-class AIPW effects and causal diagnostics |
| `05_prediction.py` | Optional supplementary prediction benchmark |
| `06_results.py` | Build final summaries and the main figure panel |

## Project Layout

```text
ML4H/
├── 01_download_data.py
├── 02_preprocess.py
├── 03_eda.py
├── 04_causal_analysis.py
├── 05_prediction.py
├── 06_results.py
├── config.py
├── drug_classes.py
├── data/
│   ├── raw/
│   └── processed/
└── outputs/
    ├── figures/
    ├── tables/
    └── models/
```

## Current Result Snapshot

The largest positive AIPW risk-difference estimates in the current run are for gout drugs, beta blockers, opioids, and potassium-sparing diuretics. Several classes still show imperfect post-weighting balance, especially gout drugs and diuretics, so the diagnostics in `outputs/tables/causal_diagnostics_summary.csv` should be reviewed before interpreting any estimate.
