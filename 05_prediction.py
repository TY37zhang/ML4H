import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             roc_curve, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap
from config import *

plt.style.use("seaborn-v0_8-whitegrid")


def get_feature_columns(df):
    exclude = {"SEQN", "kidney_stones", "KIQ026", "KID028", "cycle", "survey_weight",
               "WTMEC2YR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2",
               "SMQ020", "SMQ040", "DIQ010", "PAQ605", "PAQ650",
               "BPQ020", "BPQ080", "MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ220",
               "obesity_category", "drug_count_bin"}
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            if df[col].nunique() > 1:
                feature_cols.append(col)
    return feature_cols


def main():
    print("=== Loading processed data ===")
    df = pd.read_parquet(PROCESSED_DIR / "analysis_ready.parquet")
    print(f"  {len(df)} rows")

    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} features")

    X_df = df[feature_cols].copy()
    nan_cols = X_df.columns[X_df.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling {len(nan_cols)} columns with remaining NaNs: {nan_cols[:5]}...")
        X_df = X_df.fillna(X_df.median())
    X = X_df.values
    y = df["kidney_stones"].values
    feature_names = feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)} ({y_train.mean()*100:.1f}% positive)")
    print(f"  Test:  {len(X_test)} ({y_test.mean()*100:.1f}% positive)")

    print("\n=== Logistic Regression ===")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=5000, random_state=RANDOM_STATE))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_prauc = average_precision_score(y_test, lr_proba)
    print(f"  ROC-AUC: {lr_auc:.4f}")
    print(f"  PR-AUC:  {lr_prauc:.4f}")

    lr_coefs = lr_pipe.named_steps["lr"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": lr_coefs})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    coef_df.to_csv(TABLES_DIR / "logistic_coefficients.csv", index=False)
    print(f"  Top features: {coef_df.head(10)['feature'].tolist()}")

    print("\n=== XGBoost ===")
    pos_weight = (1 - y_train.mean()) / y_train.mean()
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        early_stopping_rounds=50,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    xgb_prauc = average_precision_score(y_test, xgb_proba)
    print(f"  ROC-AUC: {xgb_auc:.4f}")
    print(f"  PR-AUC:  {xgb_prauc:.4f}")
    print(f"  Best iteration: {xgb_model.best_iteration}")

    xgb_model.save_model(str(MODELS_DIR / "xgboost_model.json"))

    print("\n=== Evaluation ===")
    best_model_name = "XGBoost" if xgb_auc > lr_auc else "Logistic Regression"
    best_proba = xgb_proba if xgb_auc > lr_auc else lr_proba
    print(f"  Best model: {best_model_name}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_thresh_idx = np.argmax(f1s)
    best_threshold = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5
    y_pred = (best_proba >= best_threshold).astype(int)

    perf = {
        "Model": [best_model_name, "Logistic Regression", "XGBoost"],
        "ROC-AUC": [max(lr_auc, xgb_auc), lr_auc, xgb_auc],
        "PR-AUC": [max(lr_prauc, xgb_prauc), lr_prauc, xgb_prauc],
        "Precision": [precision_score(y_test, y_pred), None, None],
        "Recall": [recall_score(y_test, y_pred), None, None],
        "F1": [f1_score(y_test, y_pred), None, None],
        "Threshold": [best_threshold, None, None],
    }
    perf_df = pd.DataFrame(perf)
    perf_df.to_csv(TABLES_DIR / "model_performance.csv", index=False)
    print(f"\n  {classification_report(y_test, y_pred, target_names=['No Stone', 'Stone'])}")

    print("\n=== Generating plots ===")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in [("Logistic Regression", lr_proba), ("XGBoost", xgb_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in [("Logistic Regression", lr_proba), ("XGBoost", xgb_proba)]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=y_test.mean(), color="k", linestyle="--", alpha=0.5, label=f"Baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Stone", "Stone"], yticklabels=["No Stone", "Stone"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({best_model_name})")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_test, best_proba, n_bins=10)
    ax.plot(prob_pred, prob_true, "s-", label=best_model_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  ROC, PR, confusion matrix, calibration plots saved")

    print("\n=== SHAP Analysis ===")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    np.save(MODELS_DIR / "shap_values.npy", shap_values)

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=20)
    plt.title("SHAP Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP| Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(mean_abs_shap)[-5:][::-1]
    for idx in top5_idx:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names, show=False, ax=ax)
        plt.tight_layout()
        fname = feature_names[idx].replace("/", "_").replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"shap_dependence_{fname}.png", dpi=300, bbox_inches="tight")
        plt.close()

    drug_feature_idx = [i for i, f in enumerate(feature_names) if f.startswith("drug_") and f != "drug_count"]
    drug_shap = mean_abs_shap[drug_feature_idx]
    drug_names = [feature_names[i].replace("drug_", "") for i in drug_feature_idx]
    drug_shap_df = pd.DataFrame({"drug_class": drug_names, "mean_abs_shap": drug_shap})
    drug_shap_df = drug_shap_df.sort_values("mean_abs_shap", ascending=False)
    drug_shap_df.to_csv(TABLES_DIR / "drug_shap_importance.csv", index=False)

    print("  SHAP plots saved")
    print("\nPrediction pipeline complete.")


if __name__ == "__main__":
    main()
