import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import *
from drug_classes import DRUG_CLASS_MAP

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def table1(df):
    stones = df[df["kidney_stones"] == 1]
    no_stones = df[df["kidney_stones"] == 0]

    rows = []

    continuous_vars = [
        ("RIDAGEYR", "Age (years)"),
        ("BMXBMI", "BMI (kg/m²)"),
        ("LBXSCA", "Serum Calcium (mg/dL)"),
        ("LBXSUA", "Serum Uric Acid (mg/dL)"),
        ("LBXSPH", "Serum Phosphorus (mg/dL)"),
        ("LBXSC3SI", "Serum Bicarbonate (mmol/L)"),
        ("LBXSCR", "Serum Creatinine (mg/dL)"),
        ("LBXSBU", "BUN (mg/dL)"),
        ("egfr", "eGFR (mL/min/1.73m²)"),
        ("acr", "Albumin-Creatinine Ratio (mg/g)"),
        ("drug_count", "Number of Prescriptions"),
        ("INDFMPIR", "Poverty-Income Ratio"),
    ]

    for var, label in continuous_vars:
        if var not in df.columns:
            continue
        s_mean = stones[var].mean()
        s_std = stones[var].std()
        n_mean = no_stones[var].mean()
        n_std = no_stones[var].std()
        _, pval = stats.ttest_ind(stones[var].dropna(), no_stones[var].dropna())
        rows.append({
            "Variable": label,
            "Stones (n={})".format(len(stones)): f"{s_mean:.1f} ± {s_std:.1f}",
            "No Stones (n={})".format(len(no_stones)): f"{n_mean:.1f} ± {n_std:.1f}",
            "p-value": f"{pval:.4f}" if pval >= 0.0001 else "<0.0001",
        })

    binary_vars = [
        ("sex_binary", "Male (%)"),
        ("diabetes_status", "Diabetes (%)"),
        ("MCQ160B", "CHF (%)"),
        ("MCQ160C", "CHD (%)"),
        ("BPQ020", "Hypertension (%)"),
        ("BPQ080", "High Cholesterol (%)"),
        ("polypharmacy", "Polypharmacy ≥5 drugs (%)"),
    ]

    for var, label in binary_vars:
        if var not in df.columns:
            continue
        s_pct = stones[var].mean() * 100
        n_pct = no_stones[var].mean() * 100
        ct = pd.crosstab(df["kidney_stones"], df[var])
        _, pval, _, _ = stats.chi2_contingency(ct)
        rows.append({
            "Variable": label,
            "Stones (n={})".format(len(stones)): f"{s_pct:.1f}%",
            "No Stones (n={})".format(len(no_stones)): f"{n_pct:.1f}%",
            "p-value": f"{pval:.4f}" if pval >= 0.0001 else "<0.0001",
        })

    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "table1_demographics.csv", index=False)
    print("  Table 1 saved")
    return table


def drug_prevalence_table(df):
    drug_cols = [f"drug_{cls}" for cls in DRUG_CLASS_MAP if f"drug_{cls}" in df.columns]
    rows = []
    for col in drug_cols:
        cls_name = col.replace("drug_", "")
        users = df[df[col] == 1]
        non_users = df[df[col] == 0]
        n_users = len(users)
        if n_users < 20:
            continue
        stone_rate_users = users["kidney_stones"].mean()
        stone_rate_non = non_users["kidney_stones"].mean()
        rr = stone_rate_users / stone_rate_non if stone_rate_non > 0 else np.nan
        se_log_rr = np.sqrt(
            (1 - stone_rate_users) / (stone_rate_users * n_users) +
            (1 - stone_rate_non) / (stone_rate_non * len(non_users))
        ) if stone_rate_users > 0 and stone_rate_non > 0 else np.nan
        ci_low = np.exp(np.log(rr) - 1.96 * se_log_rr) if not np.isnan(se_log_rr) else np.nan
        ci_high = np.exp(np.log(rr) + 1.96 * se_log_rr) if not np.isnan(se_log_rr) else np.nan

        rows.append({
            "drug_class": cls_name,
            "n_users": n_users,
            "stone_rate_users": round(stone_rate_users * 100, 1),
            "stone_rate_non_users": round(stone_rate_non * 100, 1),
            "relative_risk": round(rr, 2),
            "rr_ci_low": round(ci_low, 2) if not np.isnan(ci_low) else np.nan,
            "rr_ci_high": round(ci_high, 2) if not np.isnan(ci_high) else np.nan,
        })

    drug_table = pd.DataFrame(rows).sort_values("relative_risk", ascending=False)
    drug_table.to_csv(TABLES_DIR / "drug_class_prevalence.csv", index=False)
    print("  Drug prevalence table saved")
    return drug_table


def plot_correlation_matrix(df):
    cont_vars = [
        "RIDAGEYR", "BMXBMI", "LBXSCA", "LBXSUA", "LBXSPH", "LBXSC3SI",
        "LBXSCR", "LBXSBU", "egfr", "acr", "systolic_bp", "diastolic_bp",
        "drug_count", "INDFMPIR"
    ]
    labels = [
        "Age", "BMI", "Calcium", "Uric Acid", "Phosphorus", "Bicarbonate",
        "Creatinine", "BUN", "eGFR", "ACR", "Systolic BP", "Diastolic BP",
        "Drug Count", "PIR"
    ]
    available = [(v, l) for v, l in zip(cont_vars, labels) if v in df.columns]
    vars_use = [v for v, l in available]
    labels_use = [l for v, l in available]

    corr = df[vars_use].corr()
    corr.index = labels_use
    corr.columns = labels_use

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation Matrix of Continuous Features")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Correlation matrix saved")


def plot_drug_stone_rates(drug_table, bg_rate):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d32f2f" if r > bg_rate else "#1976d2" for r in drug_table["stone_rate_users"]]
    bars = ax.barh(drug_table["drug_class"], drug_table["stone_rate_users"], color=colors, alpha=0.8)
    ax.axvline(x=bg_rate, color="black", linestyle="--", linewidth=1.5, label=f"Background ({bg_rate}%)")
    ax.set_xlabel("Kidney Stone Rate (%)")
    ax.set_title("Kidney Stone Rate by Drug Class")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "drug_class_stone_rates.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Drug stone rates plot saved")


def plot_distributions(df):
    vars_plot = [
        ("RIDAGEYR", "Age (years)"),
        ("BMXBMI", "BMI (kg/m²)"),
        ("LBXSCA", "Serum Calcium (mg/dL)"),
        ("LBXSUA", "Serum Uric Acid (mg/dL)"),
        ("egfr", "eGFR (mL/min/1.73m²)"),
        ("drug_count", "Number of Prescriptions"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (var, label) in enumerate(vars_plot):
        ax = axes[idx // 3, idx % 3]
        stones = df[df["kidney_stones"] == 1][var].dropna()
        no_stones = df[df["kidney_stones"] == 0][var].dropna()
        ax.hist(no_stones, bins=50, alpha=0.5, density=True, label="No Stones", color="#1976d2")
        ax.hist(stones, bins=50, alpha=0.5, density=True, label="Stones", color="#d32f2f")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle("Feature Distributions by Kidney Stone Status", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions_by_stone_status.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Distribution plots saved")


def plot_polypharmacy(df):
    df["drug_count_bin"] = pd.cut(df["drug_count"], bins=[-1, 0, 1, 2, 3, 4, 100],
                                   labels=["0", "1", "2", "3", "4", "5+"])
    rates = df.groupby("drug_count_bin")["kidney_stones"].agg(["mean", "count"]).reset_index()
    rates["se"] = np.sqrt(rates["mean"] * (1 - rates["mean"]) / rates["count"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rates["drug_count_bin"], rates["mean"] * 100, yerr=rates["se"] * 1.96,
           color="#5c6bc0", alpha=0.8, capsize=5)
    ax.set_xlabel("Number of Prescription Drugs")
    ax.set_ylabel("Kidney Stone Rate (%)")
    ax.set_title("Kidney Stone Rate by Polypharmacy Level")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "stone_rate_by_drug_count.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Polypharmacy plot saved")


def main():
    print("=== Loading processed data ===")
    df = pd.read_parquet(PROCESSED_DIR / "analysis_ready.parquet")
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("\n=== Generating Table 1 ===")
    t1 = table1(df)
    print(t1.to_string(index=False))

    print("\n=== Drug class prevalence ===")
    drug_table = drug_prevalence_table(df)
    print(drug_table.to_string(index=False))

    print("\n=== Generating plots ===")
    plot_correlation_matrix(df)
    plot_drug_stone_rates(drug_table, df["kidney_stones"].mean() * 100)
    plot_distributions(df)
    plot_polypharmacy(df)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
