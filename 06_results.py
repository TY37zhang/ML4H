import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

plt.style.use("seaborn-v0_8-whitegrid")


def executive_summary():
    ate_path = TABLES_DIR / "ate_summary.csv"
    drug_prev_path = TABLES_DIR / "drug_class_prevalence.csv"

    frames = {}
    if ate_path.exists():
        frames["ate"] = pd.read_csv(ate_path)
    if drug_prev_path.exists():
        frames["prev"] = pd.read_csv(drug_prev_path)

    if "ate" not in frames:
        print("  No ATE results found, skipping executive summary")
        return

    summary_cols = ["drug_class", "n_treated", "ate", "ci_low", "ci_high",
                    "p_value", "p_value_fdr", "significant_fdr", "e_value",
                    "max_smd_before", "max_smd_after_ipw"]
    summary_cols = [c for c in summary_cols if c in frames["ate"].columns]
    summary = frames["ate"][summary_cols].copy()

    if "prev" in frames:
        prev = frames["prev"][["drug_class", "relative_risk", "stone_rate_users"]].copy()
        prev.columns = ["drug_class", "crude_rr", "crude_stone_rate"]
        summary = summary.merge(prev, on="drug_class", how="left")

    summary = summary.sort_values("ate", ascending=False)
    summary.to_csv(TABLES_DIR / "executive_summary.csv", index=False)
    print("  Executive summary saved")
    print(summary.to_string(index=False))
    return summary


def causal_diagnostics_summary():
    ate_path = TABLES_DIR / "ate_summary.csv"
    balance_path = TABLES_DIR / "covariate_balance.csv"

    if not ate_path.exists():
        print("  Missing ATE file, skipping diagnostics")
        return

    ate = pd.read_csv(ate_path)
    cols = ["drug_class", "ate", "ci_low", "ci_high", "e_value", "significant_fdr",
            "max_smd_before", "max_smd_after_ipw"]
    cols = [c for c in cols if c in ate.columns]
    diagnostics = ate[cols].copy()

    if balance_path.exists():
        balance = pd.read_csv(balance_path)
        balance_max = balance.groupby("drug_class", as_index=False).agg(
            max_variable_smd_before=("smd_before", "max"),
            max_variable_smd_after_ipw=("smd_after_ipw", "max"),
        )
        diagnostics = diagnostics.merge(balance_max, on="drug_class", how="left")

    diagnostics = diagnostics.sort_values("ate", ascending=False)
    diagnostics.to_csv(TABLES_DIR / "causal_diagnostics_summary.csv", index=False)
    print("  Causal diagnostics summary saved")


def main_figure_panel():
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    drug_prev_path = TABLES_DIR / "drug_class_prevalence.csv"
    if drug_prev_path.exists():
        ax = axes[0, 0]
        dp = pd.read_csv(drug_prev_path).sort_values("stone_rate_users")
        colors = ["#d32f2f" if r > 9.3 else "#1976d2" for r in dp["stone_rate_users"]]
        ax.barh(dp["drug_class"], dp["stone_rate_users"], color=colors, alpha=0.8)
        ax.axvline(x=9.3, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Stone Rate (%)")
        ax.set_title("(A) Crude Stone Rates by Drug Class")

    ate_path = TABLES_DIR / "ate_summary.csv"
    if ate_path.exists():
        ax = axes[0, 1]
        ate = pd.read_csv(ate_path).sort_values("ate")
        for i, row in ate.iterrows():
            color = "#d32f2f" if row.get("significant_fdr") and row["ate"] > 0 else \
                    "#1976d2" if row.get("significant_fdr") and row["ate"] < 0 else "#9e9e9e"
            idx = list(ate.index).index(i)
            ax.errorbar(row["ate"], idx,
                       xerr=[[row["ate"] - row["ci_low"]], [row["ci_high"] - row["ate"]]],
                       fmt="o", color=color, capsize=4, markersize=7)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(ate)))
        ax.set_yticklabels(ate["drug_class"].values)
        ax.set_xlabel("ATE (Risk Difference)")
        ax.set_title("(B) Causal Effects (AIPW, FDR-corrected)")

    if ate_path.exists():
        ax = axes[1, 0]
        ate = pd.read_csv(ate_path).dropna(subset=["e_value"]).sort_values("e_value")
        colors = ["#d32f2f" if row["ate"] > 0 else "#1976d2" for _, row in ate.iterrows()]
        ax.barh(ate["drug_class"], ate["e_value"], color=colors, alpha=0.8)
        ax.axvline(x=2.0, color="black", linestyle="--", linewidth=1, label="E-value = 2")
        ax.set_xlabel("E-value")
        ax.set_title("(C) Sensitivity to Unmeasured Confounding")
        ax.legend()
    else:
        axes[1, 0].text(0.5, 0.5, "E-values\n(run 04_causal_analysis.py first)",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("(C) Sensitivity to Unmeasured Confounding")

    balance_path = TABLES_DIR / "covariate_balance.csv"
    if balance_path.exists():
        ax = axes[1, 1]
        balance = pd.read_csv(balance_path)
        balance_max = balance.groupby("drug_class", as_index=False).agg(
            smd_before=("smd_before", "max"),
            smd_after_ipw=("smd_after_ipw", "max"),
        ).sort_values("smd_after_ipw")
        y = np.arange(len(balance_max))
        ax.barh(y - 0.2, balance_max["smd_before"], height=0.4, label="Before", color="#9e9e9e")
        ax.barh(y + 0.2, balance_max["smd_after_ipw"], height=0.4, label="After IPW", color="#5c6bc0")
        ax.axvline(x=0.1, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(balance_max["drug_class"])
        ax.set_xlabel("Maximum absolute SMD")
        ax.set_title("(D) Covariate Balance Diagnostics")
        ax.legend()
    else:
        axes[1, 1].text(0.5, 0.5, "Covariate balance\n(run 04_causal_analysis.py first)",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("(D) Covariate Balance Diagnostics")

    plt.suptitle("Causal ML for Drug-Associated Kidney Stone Risk (NHANES 2007-2018)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "main_figure_panel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Main figure panel saved")


def main():
    print("=== Generating final results ===")

    print("\n--- Executive Summary ---")
    executive_summary()

    print("\n--- Causal Diagnostics ---")
    causal_diagnostics_summary()

    print("\n--- Main Figure Panel ---")
    main_figure_panel()

    print("\n=== Results compilation complete ===")
    print(f"\nAll outputs in:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Models:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
