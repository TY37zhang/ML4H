# Causal Machine Learning for Identifying Drug-Associated Kidney Stone Risk Using NHANES

## 1. Introduction

Kidney stone disease is common, painful, recurrent, and costly. In the United States, kidney stone prevalence has increased substantially over time, rising from approximately 3.2 percent in 1980 to 8.8 percent by 2010 (Scales et al., 2012). Kidney stones are also clinically heterogeneous: they may be driven by metabolic disease, diet, dehydration, urinary chemistry, genetic predisposition, kidney function, and medication exposure. Because many prescription medications alter urine volume, serum and urinary electrolytes, acid-base balance, or metabolic risk factors, prescription drug classes are plausible contributors to kidney stone risk. Examples include carbonic anhydrase inhibitors, which can promote calcium phosphate stones by increasing urinary pH and lowering citrate; loop and thiazide diuretics, which affect calcium handling; proton pump inhibitors, which may affect magnesium and mineral metabolism; and gout drugs, which are used by patients with hyperuricemia, a condition that itself increases stone risk.

The problem we address is how to estimate drug-class-level associations with kidney stone history while separating, as much as possible, medication effects from confounding by patient characteristics and underlying disease. A naive comparison of stone rates among drug users and non-users is not clinically reliable. For example, patients taking gout drugs often have high uric acid levels and gout, which independently increase the risk of uric acid stones. Similarly, patients taking antihypertensive medications may differ from non-users in age, obesity, diabetes, kidney function, and cardiovascular disease. These same covariates can influence both drug prescribing and kidney stone risk. Therefore, the core scientific question is: after adjustment for observed demographic, metabolic, clinical, lifestyle, and medication confounders, which prescription drug classes show the strongest adjusted associations with kidney stone history, and which results are most robust to measured and unmeasured confounding?

This problem is important for both clinical and public health reasons. Kidney stone disease creates substantial health care utilization through emergency department visits, imaging, urologic procedures, pain management, and recurrent monitoring. Annual U.S. health care costs related to kidney stone disease have been projected to increase by more than one billion dollars by 2030 (Ziemba and Matlaga, 2017). Drug-induced nephrolithiasis is estimated to account for only a small fraction of all stone cases, but that fraction may be underestimated because medication signals are often masked by confounding. Better population-level estimates could help clinicians identify patients who need counseling, closer monitoring, medication review, or alternative prescriptions. They could also help pharmacovigilance researchers prioritize drug classes for more rigorous longitudinal studies.

Machine learning is promising for this problem, but not primarily as a standalone prediction tool. The objective of this project is causal analysis under observational-data assumptions, not just predicting which patients have kidney stone history. Machine learning is useful because flexible models can serve as nuisance estimators inside doubly robust causal estimators. In particular, outcome models can learn nonlinear relationships between covariates and kidney stone history, while propensity score models can estimate the probability of drug-class exposure conditional on observed confounders. This supports more flexible confounding adjustment than a single parametric regression model. Machine learning also supports model interpretation: feature importance from the propensity and outcome nuisance models helps identify which covariates drive adjustment for each drug class. Finally, stratified treatment-effect analyses can explore whether adjusted effects differ by sex, age group, or diabetes status.

Our major contributions are fourfold. First, we define a target-trial-style framework with an explicit time zero: the NHANES household interview and prescription medication inventory date. Second, we estimate drug-class-level adjusted prevalence effects using augmented inverse probability weighting (AIPW), combining propensity and outcome modeling. Third, we evaluate each causal result using confidence intervals, Benjamini-Hochberg false discovery rate correction, E-values, and covariate balance diagnostics, rather than reporting ATEs in isolation. Fourth, we add sensitivity analyses based on medication duration (`RXDDAYS >= 365` and `RXDDAYS >= 730`) to test whether results change when current exposure is restricted to longer-duration users. Throughout the report, we emphasize the key limitation that NHANES is cross-sectional: it provides a clear time zero for measuring current medications and covariates, but it does not provide a prospective two-year follow-up window for incident kidney stones.

## 2. Related Work

Prior work on drug-associated kidney stones comes from several related literatures: kidney stone epidemiology, pharmacovigilance, machine learning for stone prediction, and causal machine learning.

Kidney stone epidemiology using NHANES has established that stone prevalence is associated with demographic and metabolic factors. Scales et al. (2012) used NHANES 2007-2010 to estimate modern U.S. kidney stone prevalence and showed strong associations with obesity, diabetes, sex, and age. That work demonstrated that NHANES is an appropriate data source for population-level kidney stone research, but it focused on descriptive epidemiology rather than medication-specific causal adjustment. NHANES is valuable because it links questionnaire responses, laboratory measurements, prescription medication inventories, body measures, and demographics in a nationally representative sample.

Pharmacovigilance studies have identified medication signals for nephrolithiasis, but they often rely on spontaneous adverse event reporting systems. FAERS-based studies can detect disproportionate reporting patterns, but they lack a true denominator population and generally cannot adjust for rich patient-level confounding. Kreimeyer et al. (2021) applied machine learning to causality assessment in adverse event reports, but the task was report-level classification rather than estimating population-level drug effects. Schuemie et al. (2025) emphasized the need for formal causal frameworks in pharmacovigilance because disproportionality analyses can be distorted by reporting bias, confounding by indication, and missing clinical context. These limitations motivate our use of a population survey with measured covariates and explicit causal estimands.

There is also a literature on machine learning for kidney stone prediction. Paranjpe et al. (2023) trained LASSO, Random Forest, and XGBoost models for kidney stone recurrence prediction using electronic health record data and reported AUCs in the range of approximately 0.585 to 0.618. Salehi et al. (2024) compared multiple machine learning algorithms for symptomatic stone prediction and found XGBoost performed best, with an AUC around 0.60. These studies show that kidney stone prediction is difficult, even with richer clinical data than NHANES. In our project, a supplementary prediction benchmark similarly achieved modest discrimination (XGBoost ROC-AUC 0.644 and PR-AUC 0.162). This supports our decision to make causal estimation, not standalone prediction, the primary experiment.

Causal forests and heterogeneous treatment effect methods provide another relevant foundation. Wager and Athey (2018) developed causal forests for estimating conditional average treatment effects with asymptotic guarantees. Brand et al. (2023) provided practical guidance for using causal forests in epidemiology and emphasized the importance of overlap, covariate balance, and careful interpretation. Although full causal forest estimation remains a future direction for this project, our subgroup analysis follows the same motivation: average effects may hide clinically meaningful heterogeneity across age, sex, and diabetes status.

The current gaps are clear. First, pharmacovigilance signal detection studies can identify candidate drugs but generally cannot distinguish medication effects from confounding using rich individual-level covariates. Second, most existing studies examine one drug or one narrow drug family at a time, making it difficult to compare drug classes under a common adjustment framework. Third, prediction models for kidney stones often focus on discrimination rather than causal interpretation. Fourth, few studies combine drug-class ATEs with E-values and balance diagnostics in a single analysis. Our study addresses these gaps by estimating adjusted drug-class effects across 12 major medication classes using a unified AIPW framework, reporting robustness diagnostics together with the estimated effects.

## 3. Methods

### 3.1 Target-Trial-Style Framework

We structure the experiment as a target-trial-style observational analysis. The target trial is not actually randomized, but defining it helps make the causal question concrete.

The eligibility criteria are adults aged 20 years or older in NHANES 2007-2018 with valid responses to the kidney stone question `KIQ026`. The time zero is the NHANES household interview and prescription medication inventory date. This is when prescription medication use is recorded and when most questionnaire covariates are measured. Treatment is defined separately for each drug class: a participant is considered treated for a class if they reported use of at least one medication in that class during the past 30 days at time zero. Controls are participants not currently using that drug class. The primary outcome is lifetime self-reported kidney stone history as of time zero, based on `KIQ026` ("Have you ever had kidney stones?"). The estimand is therefore an adjusted prevalence effect, not an incident two-year risk difference.

This distinction is important. NHANES does not observe participants prospectively after the medication interview, so we cannot honestly define testing as ending two years after time zero or measure new stones during that period. To address the concern about exposure timing, we use `RXDDAYS`, the reported duration of medication use, for sensitivity analyses. We repeat exposure definitions using current users with at least 365 days and at least 730 days of reported duration. These analyses ask whether current long-duration users differ from non-users, but they do not create a true prospective outcome window.

### 3.2 Dataset and Inputs

We use NHANES cycles 2007-2008 through 2017-2018. Raw XPT files are stored under `data/raw/`, and the processed analytic dataset is written to `data/processed/analysis_ready.parquet`. The pipeline links data by respondent sequence number (`SEQN`) and cycle. The final analytic sample includes 34,679 adults with valid kidney stone responses: 3,234 participants with kidney stone history (9.3 percent) and 31,445 without.

The input components include kidney conditions (`KIQ_U`), prescription medications (`RXQ_RX`), demographics (`DEMO`), standard biochemistry (`BIOPRO`), albumin and creatinine (`ALB_CR`), body measures (`BMX`), blood pressure (`BPX`), medical conditions (`MCQ`), diabetes (`DIQ`), blood pressure questionnaire (`BPQ`), smoking (`SMQ`), sleep (`SLQ`), physical activity (`PAQ`), and optional dietary recall (`DR1TOT`). In the current local checkout, `DR1TOT` files are not present, so dietary variables are allowed to be missing and are excluded when missingness is complete. This design keeps the code reproducible with the available tracked data while documenting the limitation.

The primary medication exposure comes from `RXQ_RX`, which records whether prescription medications were used in the past month, generic drug names, medication duration in days (`RXDDAYS`), and total prescription count. We map generic drug names to 20 drug classes using substring matching in `drug_classes.py`. The primary causal analysis focuses on 12 drug classes with adequate sample size and clinical relevance: loop diuretics, thiazides, proton pump inhibitors, antiepileptics, gout drugs, potassium-sparing diuretics, opioids, NSAIDs, statins, ACE inhibitors, beta blockers, and metformin.

### 3.3 Data Preprocessing and Feature Engineering

The preprocessing script `02_preprocess.py` loads all available XPT files, filters to valid kidney stone responses, and merges components within each NHANES cycle. Participants with refused or unknown kidney stone responses are excluded. The binary outcome `kidney_stones` equals 1 for `KIQ026 == 1` and 0 for `KIQ026 == 2`. The script also creates time-zero metadata columns, including the NHANES cycle label and the prescription exposure window of 30 days.

We derive clinical features that are plausibly related to kidney stone history or medication prescribing. These include estimated glomerular filtration rate (eGFR) using the CKD-EPI 2021 creatinine equation, albumin-creatinine ratio (ACR), calcium-phosphorus ratio, BUN-creatinine ratio, sodium-potassium ratio, BUN-eGFR ratio, pulse pressure, anion gap, and several interaction terms such as age by BMI, drug count by age, uric acid by BMI, and eGFR by age. We also create indicators for sex, race/ethnicity categories, smoking status, diabetes status, physical activity, obesity category, polypharmacy (five or more prescription medications), any diuretic use, metabolic syndrome score, and high-risk drug count.

Missing values are handled in two stages. For correlated laboratory variables and derived kidney function measures, we use iterative imputation (`IterativeImputer`) to preserve relationships among lab measures. For other numeric variables, we use median imputation. When a variable has nontrivial but less than 50 percent missingness, a missingness indicator is retained. Variables with at least 50 percent missingness are excluded from imputation and downstream causal modeling. The current processed dataset contains 226 columns after feature engineering, drug-duration features, and missingness indicators.

### 3.4 Causal Model Selection

The primary estimator is augmented inverse probability weighting (AIPW). AIPW is doubly robust because it combines two nuisance models: a propensity model for treatment assignment and an outcome model for kidney stone history. If either the propensity model or the outcome model is correctly specified, the AIPW estimator can remain consistent under standard causal assumptions.

For each drug class, the propensity model estimates the probability of current drug-class use conditional on observed confounders. We use L2-penalized logistic regression with standardized covariates. The outcome model estimates the expected kidney stone outcome conditional on confounders among treated and control participants. We use gradient boosting regressors as flexible nuisance outcome models. Propensity scores are clipped to avoid extreme weights. Other current-use drug class indicators are included as confounders, excluding the target drug class. Duration-derived indicators are not used as confounders in the primary model to avoid contaminating the current-use exposure definition.

The confounder set includes age, sex, race/ethnicity, BMI, waist circumference, poverty-income ratio, serum calcium, uric acid, phosphorus, bicarbonate, BUN, creatinine, eGFR, ACR, systolic and diastolic blood pressure, smoking status, diabetes status, physical activity, CHF, CHD, hypertension, high cholesterol, drug count, and other drug class indicators. This set is intended to adjust for demographic, metabolic, kidney function, cardiovascular, lifestyle, and medication burden differences that influence both prescribing and kidney stone history.

### 3.5 Model Evaluation and Interpretation

The main causal outputs are ATE, standard error, 95 percent confidence interval, p-value, Benjamini-Hochberg FDR-adjusted p-value, E-value, and covariate balance. The ATE is interpreted as an adjusted absolute percentage-point difference in kidney stone history for current users versus non-users of the drug class. E-values are reported alongside ATEs. The E-value represents the minimum strength of association, on a risk-ratio scale, that an unmeasured confounder would need to have with both drug exposure and kidney stone history to explain away the observed estimate. Larger E-values suggest greater robustness to unmeasured confounding, but they do not prove causality.

Covariate balance is evaluated using standardized mean differences before and after IPW. The output `covariate_balance.csv` reports before/after balance by drug class and covariate, and `causal_diagnostics_summary.csv` reports maximum SMDs by drug class. We also generate propensity overlap plots for each drug class. For multiple testing across 12 drug classes, p-values are adjusted using the Benjamini-Hochberg false discovery rate procedure.

For interpretation, we do not treat prediction as a separate experimental stage. Instead, we extract feature importance from the nuisance propensity and outcome models used inside AIPW. These feature importance outputs are saved in `causal_feature_importance.csv` and used to understand which covariates drive adjustment. A supplementary prediction benchmark using logistic regression and XGBoost is retained only to show the limited discriminative performance of cross-sectional NHANES features.

## 4. Results

### 4.1 Cohort Characteristics

The final sample includes 34,679 adults, of whom 3,234 reported a kidney stone history. Stone formers were older, more likely male, had higher BMI, had lower eGFR, and had higher prevalence of diabetes, hypertension, high cholesterol, CHF, CHD, and polypharmacy. These patterns match known kidney stone epidemiology and support the need for careful confounding adjustment.

**Table 1. Selected cohort characteristics by kidney stone history**

| Variable | Stones (n = 3,234) | No Stones (n = 31,445) | p-value |
|---|---:|---:|---:|
| Age (years) | 56.1 +/- 16.3 | 49.3 +/- 17.9 | <0.0001 |
| Male | 55.1 percent | 47.8 percent | <0.0001 |
| BMI (kg/m2) | 30.4 +/- 6.9 | 29.0 +/- 6.8 | <0.0001 |
| Serum uric acid (mg/dL) | 5.6 +/- 1.5 | 5.4 +/- 1.4 | <0.0001 |
| eGFR (mL/min/1.73m2) | 88.6 +/- 22.7 | 95.0 +/- 21.6 | <0.0001 |
| Diabetes | 29.1 percent | 16.8 percent | <0.0001 |
| Hypertension | 51.1 percent | 34.8 percent | <0.0001 |
| High cholesterol | 44.5 percent | 31.9 percent | <0.0001 |
| Polypharmacy >= 5 drugs | 31.4 percent | 16.7 percent | <0.0001 |
| Number of prescriptions | 3.5 +/- 3.6 | 2.1 +/- 2.8 | <0.0001 |

### 4.2 Primary Causal Effects

The AIPW results are summarized in Table 2 and visualized in the forest plot. Six drug classes showed significant positive adjusted effects after FDR correction: gout drugs, beta blockers, opioids, potassium-sparing diuretics, thiazides, and metformin. Four drug classes showed significant negative adjusted effects: NSAIDs, ACE inhibitors, loop diuretics, and statins. PPIs and antiepileptics were not FDR-significant in the updated causal run.

**Figure 1. AIPW forest plot of drug-class effects**

![Forest plot of AIPW estimates](outputs/figures/forest_plot_ate.png)

**Table 2. AIPW average treatment effects, E-values, and FDR significance**

| Drug class | Users | ATE | 95 percent CI | FDR significant | E-value |
|---|---:|---:|---:|---:|---:|
| Gout drugs | 564 | +5.4 percent | [4.3, 6.4] | Yes | 2.53 |
| Beta blockers | 4,439 | +5.2 percent | [4.0, 6.4] | Yes | 2.48 |
| Opioids | 2,237 | +3.5 percent | [2.4, 4.5] | Yes | 2.09 |
| Potassium-sparing | 720 | +2.4 percent | [1.5, 3.2] | Yes | 1.82 |
| Thiazides | 3,479 | +1.2 percent | [0.2, 2.3] | Yes | 1.52 |
| Metformin | 2,870 | +1.0 percent | [0.1, 1.9] | Yes | 1.46 |
| PPIs | 3,265 | +0.9 percent | [-0.1, 1.9] | No | 1.42 |
| Antiepileptics | 1,638 | -0.7 percent | [-1.7, 0.4] | No | 1.36 |
| NSAIDs | 1,874 | -1.2 percent | [-2.1, -0.3] | Yes | 1.56 |
| ACE inhibitors | 4,694 | -1.3 percent | [-2.3, -0.2] | Yes | 1.58 |
| Loop diuretics | 1,303 | -1.7 percent | [-2.6, -0.8] | Yes | 1.73 |
| Statins | 6,695 | -1.8 percent | [-2.9, -0.7] | Yes | 1.78 |

The largest positive adjusted estimates were for gout drugs and beta blockers. Gout drugs had an ATE of +5.4 percentage points and an E-value of 2.53. This result is clinically plausible as a marker of high-risk patients, but it is also highly vulnerable to confounding by indication because gout and hyperuricemia are themselves stone risk factors. Beta blockers had a similar positive ATE (+5.2 percentage points) and E-value (2.48), which may reflect cardiovascular and metabolic disease burden not fully captured by observed covariates. Opioids showed a +3.5 percentage point adjusted effect with an E-value of 2.09, potentially reflecting pain, comorbidity, or unmeasured health care utilization.

The negative estimates for statins, loop diuretics, ACE inhibitors, and NSAIDs should also be interpreted cautiously. Some of these results may reflect residual confounding, selection, or model behavior rather than a protective biological effect. For example, the loop diuretic result differs between the primary AIPW analysis and some duration sensitivity analyses, suggesting instability and confounding by cardiovascular or kidney disease severity.

### 4.3 Causal Diagnostics

The final result panel combines crude drug class stone rates, AIPW estimates, E-values, and covariate balance diagnostics.

**Figure 2. Main causal result panel**

![Main causal panel](outputs/figures/main_figure_panel.png)

Covariate balance improved after IPW for many drug classes, but maximum post-IPW SMDs remained high for several classes. For example, loop diuretics had a maximum post-IPW SMD of 1.132, potassium-sparing diuretics 0.963, gout drugs 0.928, thiazides 0.777, and metformin 0.758. These values indicate that treatment and control groups remain meaningfully different on some covariates even after weighting. Therefore, the ATE results should be viewed as adjusted observational estimates rather than definitive causal effects.

**Table 3. Selected causal diagnostics**

| Drug class | ATE | E-value | Max SMD before | Max SMD after IPW |
|---|---:|---:|---:|---:|
| Gout drugs | +5.4 percent | 2.53 | 1.574 | 0.928 |
| Beta blockers | +5.2 percent | 2.48 | 1.553 | 0.400 |
| Opioids | +3.5 percent | 2.09 | 1.205 | 0.353 |
| Potassium-sparing | +2.4 percent | 1.82 | 1.196 | 0.963 |
| Metformin | +1.0 percent | 1.46 | 2.459 | 0.758 |
| Loop diuretics | -1.7 percent | 1.73 | 2.127 | 1.132 |
| Statins | -1.8 percent | 1.78 | 1.573 | 0.611 |

The E-values help quantify sensitivity to unmeasured confounding. Gout drugs, beta blockers, and opioids had E-values above 2.0, meaning an unmeasured confounder would need a risk-ratio association above approximately 2 with both exposure and outcome to fully explain away the point estimate. However, E-values must be interpreted in clinical context. For gout drugs, such a confounder is plausible because gout and hyperuricemia are closely related to stone formation. Thus, the E-value does not eliminate concern about confounding by indication.

### 4.4 Duration Sensitivity

Duration sensitivity analyses redefined exposure as current drug class use with `RXDDAYS >= 365` or `RXDDAYS >= 730`. These analyses were implemented as IPW sensitivity checks rather than the primary AIPW estimator. They are useful for examining whether longer-duration current use changes the estimated direction or magnitude.

Several chronic medication classes showed stronger positive IPW associations in duration-restricted analyses. For example, gout drugs had ATEs of +6.9 percentage points for at least 365 days and +7.5 percentage points for at least 730 days. Beta blockers remained around +5.4 percentage points. However, some results changed direction compared with primary AIPW estimates, particularly loop diuretics and statins, which were negative in the primary AIPW analysis but positive in duration-restricted IPW sensitivity analyses. This pattern suggests that duration restrictions may increase confounding by chronic disease burden: long-term medication users are often older and sicker than non-users. Therefore, duration sensitivity should be interpreted as a stress test rather than confirmatory evidence.

### 4.5 Supplementary Prediction Benchmark

Standalone prediction was not the primary experiment because AIPW already uses prediction internally through nuisance models. Still, we retained a supplementary benchmark to understand how well the available cross-sectional features discriminate kidney stone history. Logistic regression achieved ROC-AUC 0.642 and PR-AUC 0.162. XGBoost achieved ROC-AUC 0.644 and PR-AUC 0.162, with precision 0.176, recall 0.323, and F1 0.228 at the selected threshold. These modest results are consistent with prior literature showing that kidney stone prediction is difficult and support the decision to focus the report on causal estimation rather than classification performance.

**Table 4. Supplementary prediction benchmark**

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.642 | 0.162 | NA | NA | NA |
| XGBoost | 0.644 | 0.162 | 0.176 | 0.323 | 0.228 |

## 5. Discussion

This project contributes a causal machine learning framework for evaluating drug-class associations with kidney stone history in NHANES. The main contribution is not a high-performing prediction model, but a reproducible causal analysis pipeline with explicit time zero, drug exposure definitions, AIPW estimation, E-values, balance diagnostics, subgroup outputs, and duration sensitivity analyses. This directly addresses the main weakness of the intermediate report, which mixed causal analysis and prediction without a clear estimand.

Several aspects worked well. First, the target-trial-style framing clarified what the data can and cannot support. Time zero is now explicitly defined as the NHANES medication interview date, and the outcome is clearly lifetime kidney stone history at that same time. Second, AIPW provided a structured way to combine flexible outcome modeling with propensity weighting. Third, reporting E-values alongside ATEs made the causal results more interpretable. Fourth, the balance diagnostics revealed where the analysis remains weak. Without these diagnostics, the ATE table could appear more definitive than warranted.

Several aspects did not work as well. The biggest limitation is temporal ambiguity. Because NHANES is cross-sectional, current medication use may occur after kidney stone onset. This is especially problematic for medications prescribed because of stone-related or metabolic conditions. Gout drugs are the clearest example: the positive ATE may reflect hyperuricemia and gout rather than a harmful drug effect. Alpha blockers are another example, although they were not included in the 12 primary causal drug classes because tamsulosin can be prescribed for stone passage. The second major limitation is residual imbalance. Some post-IPW SMDs remained high, indicating limited overlap between users and non-users for certain drug classes. This is common in observational medication studies because drug exposure is strongly tied to disease severity.

There are also methodological nuances. AIPW is doubly robust in theory, but it still depends on measured confounders, overlap, correct implementation, and plausible models. It cannot fix unmeasured confounding or reverse causation. E-values quantify how strong unmeasured confounding would need to be, but they do not identify whether such confounding exists. Duration sensitivity adds useful information, but because duration is measured retrospectively among current users, it may amplify differences between chronic medication users and non-users. The divergence between primary AIPW and duration-restricted IPW results for some cardiovascular drugs illustrates this problem.

Clinically, our findings should be viewed as hypothesis-generating. They can help prioritize drug classes for future longitudinal analyses, chart review, or electronic health record cohort studies. For example, gout drugs, beta blockers, and opioids showed the largest positive adjusted associations and may deserve closer investigation. However, we do not recommend changing prescribing decisions based on this cross-sectional analysis alone. A clinician should interpret these findings in the context of each patient's underlying conditions, stone history, urine chemistry, and alternative medication options.

For technology researchers, the project shows how causal machine learning can improve upon simple prediction models in clinical observational data. Rather than asking only whether a model can predict stones, the pipeline asks a more actionable question: which drug-class associations remain after adjustment, and how robust are they? The combination of ATEs, E-values, and balance diagnostics provides a more complete evidence profile than ROC-AUC alone.

Ethics and privacy implications are relatively limited because NHANES is publicly released and de-identified. However, there are still important ethical considerations. Medication-risk analyses can be misinterpreted by patients or clinicians if confounding is not emphasized. Labeling a drug class as harmful based on observational data could lead to inappropriate discontinuation or avoidance of beneficial medications. There is also a fairness concern: if certain demographic groups have different prescribing patterns, access to care, or missing laboratory data, causal estimates may reflect health system inequities rather than biological drug effects. Any real-world decision support tool based on this research would require stronger longitudinal validation and careful communication of uncertainty.

Future work should focus on four directions. First, the analysis should be replicated in longitudinal electronic health record data where medication initiation, baseline covariates, and incident stone outcomes can be ordered over time. Second, causal forest or other heterogeneous treatment effect methods could estimate patient-level variation beyond our simple sex, age, and diabetes subgroups. Third, survey weights could be incorporated to improve national representativeness. Fourth, negative and positive control analyses should be added, such as thyroid hormone as a negative control exposure and topiramate-specific analysis as a mechanistically plausible positive control. Better urine chemistry data, including urine pH, citrate, calcium, oxalate, and volume, would also improve mechanistic interpretation.

In summary, this project finds that several prescription drug classes are associated with kidney stone history after adjustment, but the strength and credibility of these associations vary. The most reliable conclusion is not that any one drug class definitively causes stones, but that a causal ML framework can organize observational evidence, reveal where confounding remains, and identify drug classes that deserve more rigorous follow-up.

## References

Brand JS, et al. (2023). Practical guide to honest causal forests for identifying heterogeneous treatment effects. American Journal of Epidemiology, 192(7):1155-1165.

Kreimeyer K, et al. (2021). Feature engineering and machine learning for causality assessment in pharmacovigilance. Computers in Biology and Medicine, 135:104517.

Paranjpe I, et al. (2023). Machine learning models to predict kidney stone recurrence using 24 hour urine testing and EHR-derived features. Journal of Urology.

Salehi A, et al. (2024). Predicting symptomatic kidney stones using machine learning algorithms. BMC Research Notes, 17:345.

Scales CD Jr, et al. (2012). Prevalence of kidney stones in the United States. European Urology, 62(1):160-165.

Schuemie MJ, et al. (2025). Causal inference tools for pharmacovigilance: using causal graphs to identify and address biases. Drug Safety.

Wager S, Athey S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. Journal of the American Statistical Association, 113(523):1228-1242.

Ziemba JB, Matlaga BR. (2017). Epidemiology and economics of nephrolithiasis. Investigative and Clinical Urology.
