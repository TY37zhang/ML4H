import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, TABLES_DIR

plt.style.use("seaborn-v0_8-whitegrid")


def get_feature_columns(df):
    exclude = {
        "SEQN", "kidney_stones", "kidney_stone_history_at_time_zero", "KIQ026", "KID028", "cycle", "survey_weight",
        "WTMEC2YR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2",
        "SMQ020", "SMQ040", "DIQ010", "PAQ605", "PAQ650",
        "BPQ020", "BPQ080", "MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ220",
        "obesity_category", "drug_count_bin",
    }
    redundant_missing = {
        "LBXSUA_missing", "LBXSPH_missing", "LBXSC3SI_missing", "LBXSBU_missing",
        "LBXSCR_missing", "LBXSNASI_missing", "LBXSKSI_missing", "LBXSCLSI_missing",
        "LBXSGL_missing", "LBXSCH_missing", "smoking_status_missing",
        "diabetes_status_missing", "physically_active_missing", "MCQ160B_missing",
        "MCQ160C_missing", "MCQ160E_missing", "MCQ160F_missing", "MCQ220_missing",
        "BPQ020_missing",
    }

    feature_cols = []
    for col in df.columns:
        if col in exclude or col in redundant_missing:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1:
            feature_cols.append(col)
    return feature_cols


def clean_feature_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], 0)

    for col in X.columns:
        cap = X[col].abs().replace([np.inf, -np.inf], np.nan).quantile(0.999)
        if pd.notna(cap) and cap > 0:
            X[col] = X[col].clip(-cap * 10, cap * 10)
    return X


def evaluate_binary_model(name, y_true, proba, include_threshold_metrics=False):
    row = {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
    }
    if include_threshold_metrics:
        precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1s)
        threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        y_pred = (proba >= threshold).astype(int)
        row.update({
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "Threshold": threshold,
        })
        return row, y_pred
    return row, None


def save_plots(y_test, model_probas, best_model_name, best_proba, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in model_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Supplementary ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in model_probas.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y=y_test.mean(), color="k", linestyle="--", alpha=0.5, label=f"Baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Supplementary Precision-Recall Curve")
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
    ax.set_title("Supplementary Calibration Plot")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    for directory in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("=== Loading processed data ===")
    df = pd.read_parquet(PROCESSED_DIR / "analysis_ready.parquet")
    print(f"  {len(df)} rows")

    feature_cols = get_feature_columns(df)
    X = clean_feature_matrix(df, feature_cols).values
    y = df["kidney_stones"].values
    print(f"  {len(feature_cols)} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("\n=== Logistic Regression ===")
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    lr_model.fit(X_train, y_train)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    print("\n=== XGBoost ===")
    pos_weight = (1 - y_train.mean()) / y_train.mean()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="auc",
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_model.save_model(str(MODELS_DIR / "xgboost_model.json"))

    lr_row, _ = evaluate_binary_model("Logistic Regression", y_test, lr_proba)
    xgb_row, xgb_pred = evaluate_binary_model("XGBoost", y_test, xgb_proba, include_threshold_metrics=True)
    perf_df = pd.DataFrame([lr_row, xgb_row])
    perf_df.to_csv(TABLES_DIR / "model_performance.csv", index=False)

    print("\n=== Supplementary Prediction Benchmark ===")
    print(perf_df.to_string(index=False))

    save_plots(
        y_test,
        {"Logistic Regression": lr_proba, "XGBoost": xgb_proba},
        "XGBoost",
        xgb_proba,
        xgb_pred,
    )
    print("\nPrediction benchmark complete.")


if __name__ == "__main__":
    main()
