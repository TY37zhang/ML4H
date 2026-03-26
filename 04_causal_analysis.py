import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from config import *

plt.style.use("seaborn-v0_8-whitegrid")

CONFOUNDER_COLS = [
    "RIDAGEYR", "sex_binary", "RIDRETH1", "BMXBMI", "BMXWAIST",
    "INDFMPIR", "LBXSCA", "LBXSUA", "LBXSPH", "LBXSC3SI", "LBXSBU",
    "LBXSCR", "egfr", "acr", "systolic_bp", "diastolic_bp",
    "smoking_status", "diabetes_status", "physically_active",
    "MCQ160B", "MCQ160C", "BPQ020", "BPQ080", "drug_count",
]

EFFECT_MODIFIER_COLS = [
    "RIDAGEYR", "sex_binary", "BMXBMI", "egfr", "LBXSCA",
    "LBXSUA", "diabetes_status", "drug_count",
]

DRUG_CLASSES_TO_ANALYZE = [
    "loop_diuretic", "thiazide", "ppi", "antiepileptic", "gout_drug",
    "potassium_sparing", "opioid", "nsaid", "statin", "ace_inhibitor",
    "beta_blocker", "metformin",
]


def get_confounders_for_class(df, target_class):
    other_drugs = [c for c in df.columns if c.startswith("drug_")
                   and c != "drug_count" and c != f"drug_{target_class}"]
    return [c for c in CONFOUNDER_COLS + other_drugs if c in df.columns]


def compute_propensity_scores(df, treatment_col, confounder_cols):
    X = df[confounder_cols].fillna(0).values
    T = df[treatment_col].values
    model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X, T)
    ps = model.predict_proba(X)[:, 1]
    return ps


def compute_smd(treated, control, var):
    t_vals = treated[var].dropna()
    c_vals = control[var].dropna()
    if len(t_vals) == 0 or len(c_vals) == 0:
        return np.nan
    pooled_std = np.sqrt((t_vals.std()**2 + c_vals.std()**2) / 2)
    if pooled_std == 0:
        return 0
    return abs(t_vals.mean() - c_vals.mean()) / pooled_std


def ipw_ate(df, treatment_col, outcome_col, ps):
    T = df[treatment_col].values
    Y = df[outcome_col].values
    ps_clipped = np.clip(ps, 0.01, 0.99)
    w1 = T / ps_clipped
    w0 = (1 - T) / (1 - ps_clipped)
    ate = np.mean(w1 * Y) / np.mean(w1) - np.mean(w0 * Y) / np.mean(w0)
    n = len(Y)
    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)
    phi = w1 * (Y - mu1) - w0 * (Y - mu0)
    se = np.std(phi) / np.sqrt(n)
    return ate, se


def aipw_ate(df, treatment_col, outcome_col, confounder_cols, ps):
    T = df[treatment_col].values
    Y = df[outcome_col].values
    X = df[confounder_cols].values
    ps_clipped = np.clip(ps, 0.01, 0.99)

    treated_idx = T == 1
    control_idx = T == 0

    outcome_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE
    )
    if treated_idx.sum() > 10 and control_idx.sum() > 10:
        outcome_model.fit(X[treated_idx], Y[treated_idx])
        mu1_hat = outcome_model.predict(X)
        outcome_model.fit(X[control_idx], Y[control_idx])
        mu0_hat = outcome_model.predict(X)
    else:
        return np.nan, np.nan

    aipw1 = mu1_hat + T / ps_clipped * (Y - mu1_hat)
    aipw0 = mu0_hat + (1 - T) / (1 - ps_clipped) * (Y - mu0_hat)

    ate = np.mean(aipw1 - aipw0)
    se = np.std(aipw1 - aipw0) / np.sqrt(len(Y))
    return ate, se


def compute_evalue(rr):
    if rr <= 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def plot_propensity_overlap(ps, treatment, class_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ps[treatment == 0], bins=50, alpha=0.5, density=True, label="Control", color="#1976d2")
    ax.hist(ps[treatment == 1], bins=50, alpha=0.5, density=True, label="Treated", color="#d32f2f")
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Propensity Score Overlap: {class_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"propensity_overlap_{class_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def analyze_drug_class(df, class_name):
    treatment_col = f"drug_{class_name}"
    if treatment_col not in df.columns:
        return None

    n_treated = df[treatment_col].sum()
    n_control = len(df) - n_treated
    if n_treated < DRUG_MIN_USERS:
        print(f"  Skipping {class_name}: only {n_treated} users")
        return None

    confounder_cols = get_confounders_for_class(df, class_name)

    print(f"  {class_name}: {n_treated} treated, {n_control} control")

    ps = compute_propensity_scores(df, treatment_col, confounder_cols)
    plot_propensity_overlap(ps, df[treatment_col].values, class_name)

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    smd_before = {v: compute_smd(treated, control, v) for v in CONFOUNDER_COLS if v in df.columns}
    max_smd = max(smd_before.values()) if smd_before else 0

    ipw_est, ipw_se = ipw_ate(df, treatment_col, "kidney_stones", ps)
    aipw_est, aipw_se = aipw_ate(df, treatment_col, "kidney_stones", confounder_cols, ps)

    use_est = aipw_est if not np.isnan(aipw_est) else ipw_est
    use_se = aipw_se if not np.isnan(aipw_se) else ipw_se

    z = use_est / use_se if use_se > 0 else 0
    pval = 2 * (1 - norm.cdf(abs(z)))

    ci_low = use_est - 1.96 * use_se
    ci_high = use_est + 1.96 * use_se

    bg_rate = df["kidney_stones"].mean()
    rr = (bg_rate + use_est) / bg_rate if bg_rate > 0 else 1
    evalue = compute_evalue(rr) if rr != 1 else np.nan

    cate_by_sex = {}
    for sex_val, sex_label in [(1, "Male"), (0, "Female")]:
        subset = df[df["sex_binary"] == sex_val]
        if len(subset) < 100:
            continue
        ps_sub = compute_propensity_scores(subset, treatment_col, confounder_cols)
        sub_est, sub_se = ipw_ate(subset, treatment_col, "kidney_stones", ps_sub)
        cate_by_sex[sex_label] = {"cate": round(sub_est, 4), "se": round(sub_se, 4)}

    cate_by_age = {}
    for label, low, high in [("20-44", 20, 44), ("45-64", 45, 64), ("65+", 65, 200)]:
        subset = df[(df["RIDAGEYR"] >= low) & (df["RIDAGEYR"] <= high)]
        if len(subset) < 100 or subset[treatment_col].sum() < 20:
            continue
        ps_sub = compute_propensity_scores(subset, treatment_col, confounder_cols)
        sub_est, sub_se = ipw_ate(subset, treatment_col, "kidney_stones", ps_sub)
        cate_by_age[label] = {"cate": round(sub_est, 4), "se": round(sub_se, 4)}

    cate_by_diabetes = {}
    for d_val, d_label in [(1, "Diabetic"), (0, "Non-diabetic")]:
        subset = df[df["diabetes_status"] == d_val]
        if len(subset) < 100 or subset[treatment_col].sum() < 20:
            continue
        ps_sub = compute_propensity_scores(subset, treatment_col, confounder_cols)
        sub_est, sub_se = ipw_ate(subset, treatment_col, "kidney_stones", ps_sub)
        cate_by_diabetes[d_label] = {"cate": round(sub_est, 4), "se": round(sub_se, 4)}

    return {
        "drug_class": class_name,
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "ipw_ate": round(ipw_est, 4),
        "ipw_se": round(ipw_se, 4),
        "aipw_ate": round(aipw_est, 4) if not np.isnan(aipw_est) else None,
        "aipw_se": round(aipw_se, 4) if not np.isnan(aipw_se) else None,
        "ate": round(use_est, 4),
        "se": round(use_se, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "p_value": round(pval, 6),
        "rr": round(rr, 3),
        "e_value": round(evalue, 2) if not np.isnan(evalue) else None,
        "max_smd_before": round(max_smd, 3),
        "cate_by_sex": cate_by_sex,
        "cate_by_age": cate_by_age,
        "cate_by_diabetes": cate_by_diabetes,
    }


def fdr_correction(results):
    pvals = [r["p_value"] for r in results]
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = pvals[idx] * n / (rank + 1)
    adjusted = np.minimum.accumulate(adjusted[sorted_idx[::-1]])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    for i, idx in enumerate(sorted_idx):
        results[idx]["p_value_fdr"] = round(adjusted[i], 6)
        results[idx]["significant_fdr"] = adjusted[i] < 0.05
    return results


def plot_forest(results):
    results_sorted = sorted(results, key=lambda x: x["ate"])
    fig, ax = plt.subplots(figsize=(10, 8))
    y_positions = range(len(results_sorted))

    for i, r in enumerate(results_sorted):
        color = "#d32f2f" if r.get("significant_fdr") and r["ate"] > 0 else \
                "#1976d2" if r.get("significant_fdr") and r["ate"] < 0 else "#9e9e9e"
        ax.errorbar(r["ate"], i, xerr=[[r["ate"] - r["ci_low"]], [r["ci_high"] - r["ate"]]],
                    fmt="o", color=color, capsize=5, markersize=8)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([r["drug_class"] for r in results_sorted])
    ax.set_xlabel("Average Treatment Effect (Risk Difference)")
    ax.set_title("Causal Effect of Drug Classes on Kidney Stone Risk\n(AIPW Estimates, BH-FDR corrected)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "forest_plot_ate.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Forest plot saved")


def main():
    print("=== Loading processed data ===")
    df = pd.read_parquet(PROCESSED_DIR / "analysis_ready.parquet")
    print(f"  {len(df)} rows")

    available_confounders = [c for c in CONFOUNDER_COLS if c in df.columns]
    print(f"  {len(available_confounders)} confounders available")

    print("\n=== Analyzing drug classes ===")
    results = []
    for cls in DRUG_CLASSES_TO_ANALYZE:
        result = analyze_drug_class(df, cls)
        if result:
            results.append(result)

    if not results:
        print("No results to report.")
        return

    results = fdr_correction(results)

    ate_df = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ["cate_by_sex", "cate_by_age", "cate_by_diabetes"]}
                          for r in results])
    ate_df = ate_df.sort_values("ate", ascending=False)
    ate_df.to_csv(TABLES_DIR / "ate_summary.csv", index=False)
    print("\n=== ATE Summary ===")
    print(ate_df[["drug_class", "n_treated", "ate", "ci_low", "ci_high", "p_value", "p_value_fdr", "significant_fdr", "e_value"]].to_string(index=False))

    subgroup_rows = []
    for r in results:
        for subgroup_type, subgroup_data in [("sex", r["cate_by_sex"]),
                                               ("age", r["cate_by_age"]),
                                               ("diabetes", r["cate_by_diabetes"])]:
            for subgroup_name, vals in subgroup_data.items():
                subgroup_rows.append({
                    "drug_class": r["drug_class"],
                    "subgroup_type": subgroup_type,
                    "subgroup": subgroup_name,
                    "cate": vals["cate"],
                    "se": vals["se"],
                })
    if subgroup_rows:
        subgroup_df = pd.DataFrame(subgroup_rows)
        subgroup_df.to_csv(TABLES_DIR / "cate_subgroups.csv", index=False)
        print("\n=== Subgroup CATEs saved ===")

    plot_forest(results)

    print("\nCausal analysis complete.")


if __name__ == "__main__":
    main()
