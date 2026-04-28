# Causal Machine Learning for Identifying Drug-Associated Kidney Stone Risk Using NHANES

We use NHANES 2007-2018 data to estimate drug-class effects on kidney stone history using a target-trial-style causal framework with augmented inverse probability weighting (AIPW), E-values, covariate balance diagnostics, and duration-based exposure sensitivity analyses.

## Dataset

- **Source**: [NHANES](https://wwwn.cdc.gov/nchs/nhanes/Default.aspx) (National Health and Nutrition Examination Survey), cycles 2007-2018
- **Sample**: 34,679 adults (20+) with valid kidney stone questionnaire responses
- **Outcome**: Self-reported lifetime kidney stone history as of the NHANES interview (KIQ026) — 3,234 cases (9.3%)
- **Features**: 226 processed columns after feature engineering, missingness indicators, 20 current drug class indicators, and drug-duration sensitivity indicators; the causal model uses demographics, labs, comorbidities, lifestyle, and other drug class indicators as confounders

## Target-Trial-Style Estimand

- **Time zero**: NHANES household interview / prescription medication inventory date
- **Primary exposure window**: reported prescription medication use in the past 30 days at time zero
- **Treatment definition**: any current use of each drug class, analyzed one class at a time
- **Duration sensitivity**: current drug class use with `RXDDAYS >= 365` and `RXDDAYS >= 730`
- **Outcome timing**: lifetime kidney stone history measured at time zero, not incident stones during prospective follow-up
- **Interpretation**: adjusted prevalence effects/associations under a target-trial-style framework; NHANES cannot identify a true two-year incident outcome window

### NHANES Components Used

| Component | Files | Description |
|-----------|-------|-------------|
| KIQ_U | Kidney Conditions | Kidney stone outcome (KIQ026) |
| RXQ_RX | Prescription Medications | Drug names, duration, count |
| BIOPRO | Biochemistry Profile | Serum calcium, uric acid, creatinine, etc. |
| ALB_CR | Albumin & Creatinine | Urine albumin, creatinine |
| DEMO | Demographics | Age, sex, race, income, education |
| BMX | Body Measures | BMI, waist circumference |
| BPX | Blood Pressure | Systolic/diastolic BP |
| MCQ | Medical Conditions | CHF, CHD, stroke, cancer |
| DIQ | Diabetes | Diabetes status |
| BPQ | Blood Pressure Questionnaire | Hypertension, high cholesterol |
| SMQ | Smoking | Smoking status |
| SLQ | Sleep | Sleep hours |
| PAQ | Physical Activity | Vigorous/moderate activity |
| DR1TOT | Dietary Intake (Day 1) | Optional dietary intake component; preprocessing tolerates missing DR1TOT files |

## Pipeline

```
01_download_data.py  →  Download/copy NHANES XPT files; current checkout has 78 raw XPT files
02_preprocess.py     →  Merge, clean, feature engineer, MICE imputation → analysis_ready.parquet
03_eda.py            →  Descriptive stats, correlation matrix, plots
04_causal_analysis.py →  AIPW effects, E-values, balance diagnostics, duration sensitivity
05_prediction.py     →  Optional supplementary prediction benchmark
06_results.py        →  Causal executive summary, diagnostics, final figure panel
```

### Setup

```bash
pip install -r requirements.txt
brew install libomp  # macOS only, required for XGBoost
```

### Run

```bash
python 01_download_data.py
python 02_preprocess.py
python 03_eda.py
python 04_causal_analysis.py
python 05_prediction.py  # optional supplementary prediction benchmark
python 06_results.py
```

## Results

### Causal Effects of Drug Classes on Kidney Stone Risk (AIPW)

![Forest Plot](outputs/figures/forest_plot_ate.png)

| Drug Class | N Users | ATE (Risk Difference) | 95% CI | Significant (FDR) | E-value |
|---|---|---|---|---|---|
| Gout drugs | 564 | +5.4% | [4.3%, 6.4%] | Yes | 2.53 |
| Beta blockers | 4,439 | +5.2% | [4.0%, 6.4%] | Yes | 2.48 |
| Opioids | 2,237 | +3.5% | [2.4%, 4.5%] | Yes | 2.09 |
| Potassium-sparing | 720 | +2.4% | [1.5%, 3.2%] | Yes | 1.82 |
| Thiazides | 3,479 | +1.2% | [0.2%, 2.3%] | Yes | 1.52 |
| Metformin | 2,870 | +1.0% | [0.1%, 1.9%] | Yes | 1.46 |
| PPIs | 3,265 | +0.9% | [-0.1%, 1.9%] | No | 1.42 |
| Antiepileptics | 1,638 | -0.7% | [-1.7%, 0.4%] | No | 1.36 |
| NSAIDs | 1,874 | -1.2% | [-2.1%, -0.3%] | Yes | 1.56 |
| ACE inhibitors | 4,694 | -1.3% | [-2.3%, -0.2%] | Yes | 1.58 |
| Loop diuretics | 1,303 | -1.7% | [-2.6%, -0.8%] | Yes | 1.73 |
| Statins | 6,695 | -1.8% | [-2.9%, -0.7%] | Yes | 1.78 |

### Causal Diagnostics

![Main Panel](outputs/figures/main_figure_panel.png)

Key diagnostic tables:

- `outputs/tables/ate_summary.csv` — AIPW ATEs, confidence intervals, FDR correction, E-values, and max SMDs
- `outputs/tables/covariate_balance.csv` — before/after IPW standardized mean differences
- `outputs/tables/causal_diagnostics_summary.csv` — compact ATE/E-value/balance summary
- `outputs/tables/ate_duration_sensitivity.csv` — `RXDDAYS >= 365` and `RXDDAYS >= 730` sensitivity results
- `outputs/tables/causal_feature_importance.csv` — nuisance-model feature importance for causal interpretation

Balance improved after IPW for many classes, but several maximum post-IPW SMDs remain high, especially loop diuretics, potassium-sparing diuretics, gout drugs, thiazides, and metformin. These estimates should therefore be interpreted as adjusted causal estimates under assumptions, not definitive causal proof.

### Crude Drug Class Stone Rates

![Drug Class Stone Rates](outputs/figures/drug_class_stone_rates.png)

### Supplementary Predictive Model Performance

Prediction is not the primary experiment; it is retained only as a secondary benchmark because the causal nuisance models already perform prediction inside AIPW.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.644 | 0.162 | 0.176 | 0.323 | 0.228 |
| Logistic Regression | 0.642 | 0.162 | — | — | — |

The optional `05_prediction.py` script can still generate ROC/PR and SHAP artifacts, but these are not part of the primary causal estimand. Main interpretability now comes from `outputs/tables/causal_feature_importance.csv`, which summarizes the nuisance models used inside AIPW.

Supplementary prediction plots:

![ROC Curve](outputs/figures/roc_curve.png)

![PR Curve](outputs/figures/pr_curve.png)

![Calibration](outputs/figures/calibration_plot.png)

### Feature Distributions by Stone Status

![Distributions](outputs/figures/distributions_by_stone_status.png)

### Correlation Matrix

![Correlation Matrix](outputs/figures/correlation_matrix.png)

### Polypharmacy and Stone Risk

![Polypharmacy](outputs/figures/stone_rate_by_drug_count.png)

## Key Findings

1. **Gout drugs** and **beta blockers** show the largest positive adjusted effects, but gout drugs remain highly vulnerable to confounding by indication.
2. **Opioids** and **potassium-sparing diuretics** show smaller positive adjusted effects with E-values above 1.8.
3. **NSAIDs**, **ACE inhibitors**, **loop diuretics**, and **statins** show negative adjusted effects in the current AIPW run.
4. **PPIs** and **antiepileptics** are not FDR-significant after the updated causal framing and diagnostics.
5. **Covariate balance diagnostics remain important** because several drug classes still have high maximum post-IPW SMDs.

## Limitations

- NHANES is cross-sectional: kidney stone history (ever) vs current drug use — temporal ambiguity
- There is no prospective two-year outcome window; duration-based drug exposure can be tested with `RXDDAYS`, but stone incidence after time zero is not observed
- Unmeasured confounding remains possible (E-values quantify minimum confounding strength needed)
- Self-reported outcome may undercount asymptomatic stones
- Confounding by indication is severe for gout drugs and alpha blockers (tamsulosin prescribed FOR stones)
- No urine chemistry (pH, oxalate, citrate) or dietary oxalate available in NHANES
- Supplementary prediction performance reflects the inherent ceiling of cross-sectional survey data; published EHR-based models with richer longitudinal data achieve 0.70-0.78

## Project Structure

```
ML4H/
├── config.py                  # Central configuration
├── drug_classes.py            # Drug name → class mapping
├── 01_download_data.py        # Data download
├── 02_preprocess.py           # Preprocessing pipeline
├── 03_eda.py                  # Exploratory data analysis
├── 04_causal_analysis.py      # Causal inference (AIPW)
├── 05_prediction.py           # Optional supplementary prediction benchmark
├── 06_results.py              # Causal results compilation
├── requirements.txt
├── data/
│   ├── raw/                   # 78 tracked NHANES XPT files; DR1TOT is optional
│   └── processed/             # analysis_ready.parquet
└── outputs/
    ├── figures/               # All plots
    ├── tables/                # CSV summary tables
    └── models/                # Optional supplementary prediction artifacts
```
