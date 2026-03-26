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
    drug_shap_path = TABLES_DIR / "drug_shap_importance.csv"

    frames = {}
    if ate_path.exists():
        frames["ate"] = pd.read_csv(ate_path)
    if drug_prev_path.exists():
        frames["prev"] = pd.read_csv(drug_prev_path)
    if drug_shap_path.exists():
        frames["shap"] = pd.read_csv(drug_shap_path)

    if "ate" not in frames:
        print("  No ATE results found, skipping executive summary")
        return

    summary = frames["ate"][["drug_class", "n_treated", "ate", "ci_low", "ci_high",
                              "p_value", "p_value_fdr", "significant_fdr", "e_value"]].copy()

    if "prev" in frames:
        prev = frames["prev"][["drug_class", "relative_risk", "stone_rate_users"]].copy()
        prev.columns = ["drug_class", "crude_rr", "crude_stone_rate"]
        summary = summary.merge(prev, on="drug_class", how="left")

    if "shap" in frames:
        shap_df = frames["shap"].copy()
        shap_df["shap_rank"] = range(1, len(shap_df) + 1)
        shap_df = shap_df[["drug_class", "mean_abs_shap", "shap_rank"]]
        summary = summary.merge(shap_df, on="drug_class", how="left")

    summary = summary.sort_values("ate", ascending=False)
    summary.to_csv(TABLES_DIR / "executive_summary.csv", index=False)
    print("  Executive summary saved")
    print(summary.to_string(index=False))
    return summary


def predictive_vs_causal():
    ate_path = TABLES_DIR / "ate_summary.csv"
    shap_path = TABLES_DIR / "drug_shap_importance.csv"

    if not ate_path.exists() or not shap_path.exists():
        print("  Missing files for comparison, skipping")
        return

    ate = pd.read_csv(ate_path)[["drug_class", "ate", "significant_fdr"]]
    shap_df = pd.read_csv(shap_path)
    shap_df["shap_rank"] = range(1, len(shap_df) + 1)

    ate["ate_rank"] = ate["ate"].rank(ascending=False).astype(int)
    merged = ate.merge(shap_df[["drug_class", "mean_abs_shap", "shap_rank"]], on="drug_class", how="outer")
    merged = merged.sort_values("ate", ascending=False)
    merged.to_csv(TABLES_DIR / "predictive_vs_causal.csv", index=False)
    print("  Predictive vs causal comparison saved")


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

    roc_path = FIGURES_DIR / "roc_curve.png"
    shap_path = FIGURES_DIR / "shap_bar.png"

    if roc_path.exists():
        ax = axes[1, 0]
        img = plt.imread(str(roc_path))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("(C) ROC Curve")
    else:
        axes[1, 0].text(0.5, 0.5, "ROC curve\n(run 05_prediction.py first)",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("(C) ROC Curve")

    if shap_path.exists():
        ax = axes[1, 1]
        img = plt.imread(str(shap_path))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("(D) SHAP Feature Importance")
    else:
        axes[1, 1].text(0.5, 0.5, "SHAP plot\n(run 05_prediction.py first)",
                        ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("(D) SHAP Feature Importance")

    plt.suptitle("Causal ML for Drug-Associated Kidney Stone Risk (NHANES 2007-2018)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "main_figure_panel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Main figure panel saved")


def main():
    print("=== Generating final results ===")

    print("\n--- Executive Summary ---")
    executive_summary()

    print("\n--- Predictive vs Causal Comparison ---")
    predictive_vs_causal()

    print("\n--- Main Figure Panel ---")
    main_figure_panel()

    print("\n=== Results compilation complete ===")
    print(f"\nAll outputs in:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"  Models:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
