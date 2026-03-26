# Causal Machine Learning for Identifying Drug-Associated Kidney Stone Risk in NHANES

**Interim Report**

---

## 1. Introduction

### What problem are we going to solve?

Kidney stones affect approximately 1 in 11 adults in the United States, with prevalence rising from 3.2% in 1980 to 8.8% by 2010 (Scales et al., 2012). Multiple prescription drug classes have been implicated in altering kidney stone risk through various pharmacological mechanisms — carbonic anhydrase inhibitors induce metabolic acidosis, loop diuretics alter calcium handling, and proton pump inhibitors disrupt magnesium homeostasis. However, identifying which drug-stone associations represent genuine causal effects versus confounding by underlying disease remains a major challenge. Patients taking gout medications, for instance, already have elevated uric acid levels that independently cause stones, making naive association studies unreliable.

This project addresses the question: **after adjusting for a rich set of demographic, metabolic, and clinical confounders, which prescription drug classes causally increase or decrease kidney stone risk, and which patient subgroups experience the largest drug-attributable effects?**

### Why is this problem important?

Kidney stone disease imposes substantial clinical and economic burden, with annual U.S. healthcare costs projected to increase by $1.24 billion by 2030 (Ziemba & Matlaga, 2017). Drug-induced nephrolithiasis accounts for an estimated 1-2% of all stone cases, but this fraction is likely underestimated because drug-stone associations are often obscured by confounding. A recent pharmacovigilance study identified 33 drugs associated with nephrolithiasis that carried no kidney stone warning on their labels (Zhang et al., 2025). Reliable causal estimates of drug-stone effects would inform prescribing decisions, support FDA labeling, and enable personalized risk assessment — particularly for patients already at elevated baseline stone risk due to metabolic or demographic factors.

### Why is machine learning promising?

Traditional pharmacoepidemiological approaches to this problem — logistic regression, Cox proportional hazards, disproportionality analysis — rely on pre-specified model forms and struggle with high-dimensional confounding. Machine learning offers three specific advantages for this problem:

1. **Flexible confounding adjustment**: Gradient-boosted models used as nuisance estimators in doubly robust causal frameworks can capture complex nonlinear relationships between confounders and both treatment assignment and outcomes, reducing residual confounding bias.
2. **Heterogeneous treatment effect discovery**: Causal forests (Wager & Athey, 2018) can identify subgroups where drug-stone effects are concentrated without requiring pre-specification of interaction terms — revealing, for example, that a drug's lithogenic effect may be strongest in older patients with low eGFR and high baseline calcium.
3. **Interpretable feature attribution**: SHAP values provide transparent, patient-level explanations of which features drive both predicted risk and estimated causal effects, bridging the gap between predictive accuracy and clinical interpretability.

---

## 2. Related Work

### Drug-induced nephrolithiasis studies

Zhang et al. (2025) conducted the largest pharmacovigilance study of drug-induced kidney stones using the FDA Adverse Event Reporting System (FAERS), identifying 38,307 nephrolithiasis reports across 21 million total records and flagging 50 associated drugs via reporting odds ratios. Li et al. (2024) similarly analyzed the top 30 kidney-stone-associated drugs in FAERS, finding antirheumatics, parathyroid hormone analogs, and antivirals among the highest-signal classes. However, these FAERS-based studies use disproportionality analysis (ROR/PRR), which can detect statistical signals but cannot estimate causal effects due to the absence of a population denominator and limited confounder data.

### Topiramate-specific evidence

Topiramate's association with nephrolithiasis is well-established mechanistically: it inhibits renal tubular carbonic anhydrase, causing metabolic acidosis, alkaline urine pH, and hypocitraturia — conditions favoring calcium phosphate stone formation. Maalouf et al. (2014) found 10.7% stone prevalence among long-term topiramate users. The largest cohort study to date, Salka et al. (2025), analyzed 1.1 million adults and found adjusted hazard ratios of 1.22-1.58 for topiramate/zonisamide users, with dose-dependent effects strongest in younger adults.

### NHANES kidney stone epidemiology

Scales et al. (2012) established modern U.S. kidney stone prevalence estimates using NHANES 2007-2010, finding 8.8% overall prevalence with strong associations with obesity and diabetes. This work demonstrated NHANES as a viable data source for stone epidemiology, though their analysis was limited to descriptive statistics and logistic regression.

### Causal ML in pharmacovigilance

Zhao et al. (2022) reviewed the integration of causal inference and machine learning for pharmacovigilance, concluding that the field lags in adopting ML-causal inference integrated models despite their theoretical promise. Kreimeyer et al. (2021) developed ML models for causality assessment in FAERS but focused on report-level classification rather than population-level causal effect estimation. Schuemie et al. (2025) introduced DAG-based causal frameworks for addressing biases in disproportionality analysis, highlighting the need for formal causal reasoning in pharmacovigilance.

### ML for kidney stone prediction

Paranjpe et al. (2023) trained LASSO, Random Forest, and XGBoost models on EHR data to predict kidney stone recurrence, achieving AUCs of 0.585-0.618 — highlighting the inherent difficulty of stone prediction. Salehi et al. (2024) found XGBoost performed best (AUC 0.60) among five ML algorithms for predicting symptomatic stones, with serum creatinine and sleep duration among key predictors.

### Causal forests in healthcare

Wager and Athey (2018) developed the causal forest algorithm for estimating heterogeneous treatment effects, proving pointwise consistency and asymptotic normality. Applications in healthcare have included septic shock treatment (Loh et al., 2025), intensive glycemic control in diabetes (Basu et al., 2021), and various RCT secondary analyses. Brand et al. (2023) provided practical guidance on implementing causal forests in epidemiological research, noting that CATE estimation can be biased when propensity models are misspecified.

### Current gaps

1. **No causal ML study of drug-stone associations**: Existing pharmacovigilance studies use signal detection methods (ROR/PRR) that cannot separate causal effects from confounding. No study has applied doubly robust estimation or causal forests to estimate drug-attributable kidney stone risk.
2. **No multi-drug-class comparison with causal adjustment**: Studies typically examine one drug at a time. A unified framework comparing causal effects across drug classes — while adjusting for the same confounder set — enables direct comparison and clinical prioritization.
3. **No heterogeneous treatment effect analysis for drug-stone risk**: It is unknown which patient subgroups face the highest drug-attributable stone risk. Identifying these subgroups could inform personalized prescribing.
4. **Underuse of NHANES for this purpose**: NHANES provides nationally representative data with both kidney stone outcomes, prescription drug use, and rich serum chemistry — yet no study has exploited this combination for causal drug-stone analysis.

---

## 3. Methods

### How are we going to solve this problem?

We employ a three-stage analytical framework:

**Stage 1 — Causal effect estimation**: For each of 12 major drug classes, we estimate the Average Treatment Effect (ATE) on kidney stone risk using Augmented Inverse Probability Weighting (AIPW). This doubly robust estimator combines a propensity score model (probability of drug class exposure given confounders) with an outcome model (probability of kidney stones given confounders), providing consistent ATE estimates if either model is correctly specified. We use gradient-boosted models as nuisance estimators within the AIPW framework, and apply Benjamini-Hochberg false discovery rate correction for multiple testing across drug classes.

**Stage 2 — Subgroup discovery**: For each drug class with a significant ATE, we estimate Conditional Average Treatment Effects (CATEs) across patient subgroups defined by sex, age group (20-44, 45-64, 65+), and diabetes status using stratified IPW estimation. This reveals heterogeneity in drug-stone effects.

**Stage 3 — Predictive modeling with interpretability**: We train logistic regression (L1-penalized baseline) and XGBoost models to predict kidney stone risk from all available features, then apply SHAP to identify the most predictive features. Comparing SHAP-based predictive importance with causal ATE estimates reveals which drug-stone associations are driven by confounding versus genuine effects.

### How our research is novel

This is the first study to: (1) apply doubly robust causal inference to estimate drug-class-level effects on kidney stone risk using nationally representative data; (2) simultaneously compare causal effects across 12 drug classes within a unified analytical framework; (3) perform subgroup-level causal effect estimation to identify which patient profiles face the highest drug-attributable stone risk; and (4) juxtapose causal estimates with SHAP-based predictive importance to explicitly distinguish confounded from causal associations.

### Dataset introduction

We use the National Health and Nutrition Examination Survey (NHANES), pooling six two-year cycles from 2007-2008 through 2017-2018. NHANES is a nationally representative cross-sectional survey of the U.S. civilian non-institutionalized population, conducted by the National Center for Health Statistics (CDC). We link 13 data components per cycle:

- **Outcome**: Kidney Conditions questionnaire (KIQ_U) — KIQ026: "Have you ever had kidney stones?" (binary)
- **Drug exposure**: Prescription Medications (RXQ_RX) — generic drug names, duration, total drug count
- **Serum chemistry**: Standard Biochemistry Profile (BIOPRO) — calcium, uric acid, phosphorus, bicarbonate, BUN, creatinine, sodium, potassium, chloride (92%+ coverage)
- **Urine labs**: Albumin & Creatinine (ALB_CR) — urine albumin, creatinine (97% coverage)
- **Demographics**: age, sex, race/ethnicity, poverty-income ratio, education
- **Body measures**: BMI, waist circumference
- **Blood pressure**: systolic and diastolic BP (exam)
- **Comorbidities**: diabetes (DIQ), hypertension and high cholesterol (BPQ), CHF, CHD, stroke, cancer (MCQ)
- **Lifestyle**: smoking status (SMQ), physical activity (PAQ)

The final analytic sample comprises **34,679 adults aged 20+** with valid kidney stone responses: **3,234 (9.3%) with kidney stone history** and **31,445 without**. Drug exposure is classified into 20 pharmacological classes using substring matching on generic drug names, with classes ranging from 522 users (bisphosphonates) to 6,695 users (statins).

Derived features include eGFR (CKD-EPI 2021 equation), albumin-creatinine ratio, composite smoking status, and a polypharmacy indicator (5+ concurrent prescriptions). Missing values are handled via median imputation with binary missingness indicators retained as features.

### Experiments setup

- **Causal analysis**: 12 drug classes analyzed (minimum 100 users). Confounders include 24 variables: demographics (age, sex, race, income), body measures (BMI, waist), serum labs (9 analytes), derived measures (eGFR, ACR), comorbidities (diabetes, hypertension, high cholesterol, CHF, CHD), lifestyle (smoking, physical activity), and other drug class indicators. AIPW uses L2-penalized logistic regression for propensity scores and gradient-boosted regressors for outcome models. Propensity score overlap is verified visually for each drug class.
- **Prediction**: 80/20 stratified train/test split. Logistic regression (L1, C=0.1) and XGBoost (500 trees, max depth 6, class-weighted, early stopping on PR-AUC). Evaluation via ROC-AUC, PR-AUC, F1 at optimal threshold, and calibration plots. SHAP TreeExplainer applied to XGBoost for feature attribution.
- **E-values**: For each significant ATE, we compute the E-value — the minimum strength of unmeasured confounding needed to explain away the observed result — to quantify robustness.

---

## 4. Results

### Preliminary results

**Table 1: Cohort characteristics by kidney stone status**

| Variable | Stones (n=3,234) | No Stones (n=31,445) | p-value |
|---|---|---|---|
| Age (years) | 56.1 +/- 16.3 | 49.3 +/- 17.9 | <0.0001 |
| Male (%) | 55.1% | 47.8% | <0.0001 |
| BMI (kg/m2) | 30.4 +/- 6.9 | 29.0 +/- 6.8 | <0.0001 |
| Serum Calcium (mg/dL) | 9.4 +/- 0.4 | 9.4 +/- 0.3 | 0.013 |
| Serum Uric Acid (mg/dL) | 5.6 +/- 1.5 | 5.4 +/- 1.4 | <0.0001 |
| eGFR (mL/min/1.73m2) | 88.8 +/- 22.8 | 95.3 +/- 21.6 | <0.0001 |
| Diabetes (%) | 29.1% | 16.8% | <0.0001 |
| Hypertension (%) | 51.1% | 34.8% | <0.0001 |
| Polypharmacy (%) | 31.4% | 16.7% | <0.0001 |
| Number of Prescriptions | 3.5 +/- 3.6 | 2.1 +/- 2.8 | <0.0001 |

Stone formers are significantly older, more likely male, more obese, and have substantially higher rates of diabetes, hypertension, and polypharmacy.

**Table 2: Causal effect estimates (AIPW, BH-FDR corrected)**

| Drug Class | N Users | ATE (Risk Difference) | 95% CI | FDR Sig. | E-value |
|---|---|---|---|---|---|
| Gout drugs | 564 | +8.8% | [7.7%, 9.9%] | Yes | 3.30 |
| Beta blockers | 4,439 | +4.6% | [3.5%, 5.7%] | Yes | 2.35 |
| Opioids | 2,237 | +3.7% | [2.7%, 4.8%] | Yes | 2.15 |
| Thiazides | 3,479 | +2.4% | [1.3%, 3.5%] | Yes | 1.83 |
| PPIs | 3,265 | +2.2% | [1.3%, 3.1%] | Yes | 1.77 |
| Potassium-sparing | 720 | +1.9% | [1.0%, 2.9%] | Yes | 1.70 |
| Antiepileptics | 1,638 | +1.3% | [0.3%, 2.4%] | Yes | 1.54 |
| Metformin | 2,870 | +1.1% | [0.3%, 2.0%] | Yes | 1.49 |
| ACE inhibitors | 4,694 | +0.7% | [-0.4%, 1.8%] | No | — |
| NSAIDs | 1,874 | -1.0% | [-2.0%, -0.1%] | Yes | 1.50 |
| Statins | 6,695 | -1.3% | [-2.6%, 0.0%] | No | — |
| Loop diuretics | 1,303 | -2.7% | [-3.8%, -1.7%] | Yes | 2.18 |

Eight drug classes show significant positive effects (increased stone risk) and two show significant protective effects after FDR correction. Loop diuretics are the most protective (-2.7% absolute risk reduction), while gout drugs show the largest risk increase (+8.8%), though the latter is likely driven by confounding by indication.

**Predictive model performance**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (L1) | 0.642 | 0.162 |
| XGBoost | 0.644 | 0.162 |

Both models achieve moderate discriminative performance. The top SHAP features include drug count, waist circumference, urine creatinine, urine albumin, and ACR — dominated by metabolic and body composition markers rather than individual drug classes. Among drug-specific SHAP features, PPIs, opioids, and beta blockers rank highest, broadly consistent with the causal ATE estimates.

---

## 5. Discussion

### Challenges so far

1. **Cross-sectional temporal ambiguity**: NHANES asks "have you *ever* had kidney stones" alongside *current* medication use. A patient currently taking tamsulosin may have been prescribed it *because* of a prior stone, not before it. This fundamental limitation means our causal estimates should be interpreted as associations adjusted for observed confounders, not definitive causal conclusions.

2. **Confounding by indication**: Gout drugs (allopurinol, colchicine) show the largest ATE (+8.8%), but gout patients inherently have hyperuricemia — a direct cause of uric acid stones. The E-value of 3.30 suggests moderate-strength unmeasured confounding could explain this result. Similarly, the positive thiazide signal (+2.4%) contradicts clinical evidence that thiazides are protective (they reduce urinary calcium excretion); this likely reflects residual confounding by hypertension severity.

3. **Moderate predictive performance**: Both models achieve ROC-AUC around 0.64, consistent with prior literature on ML-based stone prediction (Paranjpe et al., 2023: 0.58-0.62; Salehi et al., 2024: 0.60). This reflects the inherent difficulty of predicting a multifactorial condition from cross-sectional survey data without longitudinal urine chemistry or dietary details.

4. **Missing urine chemistry**: NHANES lacks urine pH, calcium, oxalate, and citrate measurements — the direct mechanistic mediators of stone formation. This limits both the predictive models and the causal analysis, as these would serve as important confounders or mediators.

### Remaining tasks

1. **Causal forests**: Implement econml's `CausalForestDML` for full heterogeneous treatment effect estimation beyond the current stratified IPW approach, enabling data-driven subgroup discovery.
2. **Sensitivity analyses**: Restrict to cycles H-J (2013-2018) where drug indication codes are available, enabling partial control for confounding by indication. Run negative control analyses using drug classes with no plausible stone mechanism (thyroid hormone).
3. **Temporal sensitivity**: Compare results when restricting to participants aged 40+ (more likely to have established drug use before stone formation) versus all adults.
4. **Improved feature engineering**: Explore drug interaction features (e.g., concurrent thiazide + calcium supplement use) and nonlinear transformations of lab values.
5. **Survey weight integration**: Incorporate NHANES survey weights into causal estimates for nationally representative inference.

### Questions for instructor/TAs

1. Given the cross-sectional limitation, is it more appropriate to frame our causal estimates as "adjusted associations" rather than "causal effects"? How should we discuss this distinction in the final report?
2. For the causal forest implementation, should we prioritize a small number of drug classes with strong mechanistic priors (e.g., loop diuretics, antiepileptics) or continue with the broad 12-class approach?
3. The thiazide result (positive ATE, contradicting known protective mechanism) likely reflects confounding by hypertension severity. Would the TAs recommend excluding thiazides from the causal analysis, or presenting this as an illustrative example of residual confounding?

---

## References

1. Basu S, Sussman JB, Hayward RA. (2021). Generalizability of heterogeneous treatment effects based on causal forests applied to two randomized clinical trials of intensive glycemic control. *Annals of Internal Medicine*. PMC8748294.
2. Brand JS, et al. (2023). Practical guide to honest causal forests for identifying heterogeneous treatment effects. *American Journal of Epidemiology*, 192(7):1155-1165.
3. Kreimeyer K, et al. (2021). Feature engineering and machine learning for causality assessment in pharmacovigilance. *Computers in Biology and Medicine*, 135:104517.
4. Li J, et al. (2024). An exploratory study evaluated the 30 most commonly reported medications associated with kidney stones. *Frontiers in Pharmacology*, 15:1377679.
5. Loh WW, et al. (2025). Application of causal forests to randomised controlled trial data to identify heterogeneous treatment effects. *BMC Medical Research Methodology*, 25:42.
6. Maalouf NM, et al. (2014). Nephrolithiasis in topiramate users. *Urological Research*. PMC4103417.
7. Paranjpe I, et al. (2023). Machine learning models to predict kidney stone recurrence using 24 hour urine testing and EHR-derived features. *Journal of Urology*. PMC10350114.
8. Salka K, et al. (2025). Associations of topiramate and zonisamide use with kidney stones: a retrospective cohort study. *American Journal of Kidney Diseases*.
9. Salehi A, et al. (2024). Predicting symptomatic kidney stones using machine learning algorithms. *BMC Research Notes*, 17:345.
10. Scales CD Jr, et al. (2012). Prevalence of kidney stones in the United States. *European Urology*, 62(1):160-165.
11. Schuemie MJ, et al. (2025). Causal inference tools for pharmacovigilance: using causal graphs to identify and address biases. *Drug Safety*.
12. Wager S, Athey S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523):1228-1242.
13. Zhang Y, et al. (2025). Drug-induced kidney stones: a real-world pharmacovigilance study using the FAERS database. *Frontiers in Pharmacology*, 16:1511115.
14. Zhao H, et al. (2022). Machine learning in causal inference: application in pharmacovigilance. *Drug Safety*, 45:459-476.
15. Ziemba JB, Matlaga BR. (2017). Epidemiology and economics of nephrolithiasis. *Investigative and Clinical Urology*. PMC9914194.
