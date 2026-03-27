import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             roc_curve, f1_score, precision_score, recall_score,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import shap
from config import *

optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.style.use("seaborn-v0_8-whitegrid")

N_FOLDS = 5
XGB_TRIALS = 200
LGB_TRIALS = 150
CB_TRIALS = 100


def get_feature_columns(df):
    exclude = {"SEQN", "kidney_stones", "KIQ026", "KID028", "cycle", "survey_weight",
               "WTMEC2YR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2",
               "SMQ020", "SMQ040", "DIQ010", "PAQ605", "PAQ650",
               "BPQ020", "BPQ080", "MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ220",
               "obesity_category", "drug_count_bin"}

    redundant_lab_missing = {
        "LBXSUA_missing", "LBXSPH_missing", "LBXSC3SI_missing", "LBXSBU_missing",
        "LBXSCR_missing", "LBXSNASI_missing", "LBXSKSI_missing", "LBXSCLSI_missing",
        "LBXSGL_missing", "LBXSCH_missing"
    }
    near_zero_missing = {
        "smoking_status_missing", "diabetes_status_missing", "physically_active_missing",
        "MCQ160B_missing", "MCQ160C_missing", "MCQ160E_missing", "MCQ160F_missing",
        "MCQ220_missing", "BPQ020_missing"
    }
    exclude = exclude | redundant_lab_missing | near_zero_missing

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            if df[col].nunique() > 1:
                feature_cols.append(col)
    return feature_cols


def select_features_by_importance(X_train, y_train, feature_names, min_importance=0.0005):
    print("  Running feature importance selection...")
    selector = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        random_state=42, eval_metric="auc"
    )
    selector.fit(X_train, y_train)
    importances = selector.feature_importances_
    mask = importances >= min_importance
    selected = [f for f, m in zip(feature_names, mask) if m]
    dropped = [f for f, m in zip(feature_names, mask) if not m]
    print(f"  Kept {len(selected)}/{len(feature_names)} features (dropped {len(dropped)})")
    if dropped:
        print(f"  Dropped: {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
    return selected, mask


def tune_xgboost(X, y, n_trials=XGB_TRIALS, n_folds=N_FOLDS, random_state=42):
    pos_weight = (1 - y.mean()) / y.mean()

    def objective(trial):
        params = {
            "n_estimators": 2000,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
            "scale_pos_weight": pos_weight,
            "random_state": random_state,
            "eval_metric": "auc",
            "early_stopping_rounds": 80,
        }
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        auc_scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            proba = model.predict_proba(X[val_idx])[:, 1]
            auc_scores.append(roc_auc_score(y[val_idx], proba))
        return np.mean(auc_scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def tune_lightgbm(X, y, n_trials=LGB_TRIALS, n_folds=N_FOLDS, random_state=42):
    pos_weight = (1 - y.mean()) / y.mean()

    def objective(trial):
        params = {
            "n_estimators": 2000,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "scale_pos_weight": pos_weight,
            "random_state": random_state,
            "verbose": -1,
        }
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        auc_scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = lgb.LGBMClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=[(X[val_idx], y[val_idx])],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
            proba = model.predict_proba(X[val_idx])[:, 1]
            auc_scores.append(roc_auc_score(y[val_idx], proba))
        return np.mean(auc_scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def tune_catboost(X, y, n_trials=CB_TRIALS, n_folds=N_FOLDS, random_state=42):
    def objective(trial):
        params = {
            "iterations": 2000,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "auto_class_weights": "Balanced",
            "random_seed": random_state,
            "verbose": 0,
            "early_stopping_rounds": 80,
        }
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        auc_scores = []
        for train_idx, val_idx in skf.split(X, y):
            model = CatBoostClassifier(**params)
            model.fit(X[train_idx], y[train_idx],
                      eval_set=(X[val_idx], y[val_idx]), verbose=0)
            proba = model.predict_proba(X[val_idx])[:, 1]
            auc_scores.append(roc_auc_score(y[val_idx], proba))
        return np.mean(auc_scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def build_stacking_features(X_train, y_train, X_test, models, n_folds=N_FOLDS, random_state=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = np.zeros((len(X_train), len(models)))
    test_preds = np.zeros((len(X_test), len(models)))

    for m_idx, (name, model_fn) in enumerate(models.items()):
        print(f"    Stacking fold predictions for {name}...")
        test_fold_preds = np.zeros((len(X_test), n_folds))

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            model = model_fn()
            X_tr_fold, y_tr_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            if name == "xgboost":
                model.fit(X_tr_fold, y_tr_fold,
                          eval_set=[(X_val_fold, y_val_fold)], verbose=False)
            elif name == "lightgbm":
                model.fit(X_tr_fold, y_tr_fold,
                          eval_set=[(X_val_fold, y_val_fold)],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
            elif name == "catboost":
                model.fit(X_tr_fold, y_tr_fold,
                          eval_set=(X_val_fold, y_val_fold), verbose=0)
            else:
                model.fit(X_tr_fold, y_tr_fold)

            oof_preds[val_idx, m_idx] = model.predict_proba(X_val_fold)[:, 1]
            test_fold_preds[:, fold_idx] = model.predict_proba(X_test)[:, 1]

        test_preds[:, m_idx] = test_fold_preds.mean(axis=1)

    return oof_preds, test_preds


def main():
    print("=== Loading processed data ===")
    df = pd.read_parquet(PROCESSED_DIR / "analysis_ready.parquet")
    print(f"  {len(df)} rows")

    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} initial features")

    X_df = df[feature_cols].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    nan_cols = X_df.columns[X_df.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling {len(nan_cols)} columns with remaining NaNs")
        X_df = X_df.fillna(X_df.median())
    X_df = X_df.replace([np.inf, -np.inf], 0)
    for col in X_df.columns:
        col_max = X_df[col].abs().replace([np.inf, -np.inf], np.nan).quantile(0.999)
        if col_max > 0:
            X_df[col] = X_df[col].clip(-col_max * 10, col_max * 10)
    X_all = X_df.values
    assert np.isfinite(X_all).all(), "Data contains inf/nan"
    y = df["kidney_stones"].values
    feature_names_all = feature_cols

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("\n=== Feature Selection ===")
    selected_features, feat_mask = select_features_by_importance(
        X_train_all, y_train, feature_names_all, min_importance=0.001
    )
    X_train = X_train_all[:, feat_mask]
    X_test = X_test_all[:, feat_mask]
    X = X_all[:, feat_mask]
    feature_names = selected_features
    print(f"  Using {len(feature_names)} features")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")

    pos_weight = (1 - y_train.mean()) / y_train.mean()

    print("\n=== Logistic Regression (Tuned C) ===")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", solver="saga", max_iter=5000,
                                  random_state=RANDOM_STATE))
    ])
    param_grid = {"lr__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]}
    grid_search = GridSearchCV(lr_pipe, param_grid,
                               cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                               scoring="roc_auc", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    lr_pipe = grid_search.best_estimator_
    print(f"  Best C: {grid_search.best_params_['lr__C']}")
    print(f"  Best CV AUC: {grid_search.best_score_:.4f}")

    lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_prauc = average_precision_score(y_test, lr_proba)
    print(f"  Test ROC-AUC: {lr_auc:.4f}")
    print(f"  Test PR-AUC:  {lr_prauc:.4f}")

    lr_coefs = lr_pipe.named_steps["lr"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": lr_coefs})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    coef_df.to_csv(TABLES_DIR / "logistic_coefficients.csv", index=False)

    print(f"\n=== XGBoost (Optuna, {XGB_TRIALS} trials) ===")
    best_xgb_params, xgb_cv = tune_xgboost(X_train, y_train, random_state=RANDOM_STATE)

    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, **best_xgb_params,
        scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
        eval_metric="auc", early_stopping_rounds=80,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    xgb_prauc = average_precision_score(y_test, xgb_proba)
    print(f"  Test ROC-AUC: {xgb_auc:.4f}")
    print(f"  Test PR-AUC:  {xgb_prauc:.4f}")
    print(f"  Best iteration: {xgb_model.best_iteration}")
    xgb_model.save_model(str(MODELS_DIR / "xgboost_model.json"))

    print(f"\n=== LightGBM (Optuna, {LGB_TRIALS} trials) ===")
    best_lgb_params, lgb_cv = tune_lightgbm(X_train, y_train, random_state=RANDOM_STATE)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000, **best_lgb_params,
        scale_pos_weight=pos_weight, random_state=RANDOM_STATE, verbose=-1,
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_proba)
    lgb_prauc = average_precision_score(y_test, lgb_proba)
    print(f"  Test ROC-AUC: {lgb_auc:.4f}")
    print(f"  Test PR-AUC:  {lgb_prauc:.4f}")

    print(f"\n=== CatBoost (Optuna, {CB_TRIALS} trials) ===")
    best_cb_params, cb_cv = tune_catboost(X_train, y_train, random_state=RANDOM_STATE)

    cb_model = CatBoostClassifier(
        iterations=2000, **best_cb_params,
        auto_class_weights="Balanced", random_seed=RANDOM_STATE,
        verbose=0, early_stopping_rounds=80,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    cb_proba = cb_model.predict_proba(X_test)[:, 1]
    cb_auc = roc_auc_score(y_test, cb_proba)
    cb_prauc = average_precision_score(y_test, cb_proba)
    print(f"  Test ROC-AUC: {cb_auc:.4f}")
    print(f"  Test PR-AUC:  {cb_prauc:.4f}")

    print("\n=== ExtraTrees ===")
    et_model = ExtraTreesClassifier(
        n_estimators=500, max_depth=12, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    et_proba = et_model.predict_proba(X_test)[:, 1]
    et_auc = roc_auc_score(y_test, et_proba)
    et_prauc = average_precision_score(y_test, et_proba)
    print(f"  Test ROC-AUC: {et_auc:.4f}")
    print(f"  Test PR-AUC:  {et_prauc:.4f}")

    print("\n=== Augmented Stacking Ensemble ===")
    stacking_models = {
        "lr": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l1", C=grid_search.best_params_["lr__C"],
                solver="saga", max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "xgboost": lambda: xgb.XGBClassifier(
            n_estimators=2000, **best_xgb_params,
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
            eval_metric="auc", early_stopping_rounds=80,
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=2000, **best_lgb_params,
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE, verbose=-1,
        ),
        "catboost": lambda: CatBoostClassifier(
            iterations=2000, **best_cb_params,
            auto_class_weights="Balanced", random_seed=RANDOM_STATE,
            verbose=0, early_stopping_rounds=80,
        ),
        "extratrees": lambda: ExtraTreesClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    oof_preds, test_stack_preds = build_stacking_features(
        X_train, y_train, X_test, stacking_models, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    top_k = min(20, X_train.shape[1])
    importances = xgb_model.feature_importances_
    top_idx = np.argsort(importances)[-top_k:]

    oof_aug = np.hstack([oof_preds, X_train_scaled[:, top_idx]])
    test_aug = np.hstack([test_stack_preds, X_test_scaled[:, top_idx]])

    print("  Training augmented meta-learner...")
    meta_model = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    meta_model.fit(oof_aug, y_train)
    stack_proba = meta_model.predict_proba(test_aug)[:, 1]
    stack_auc = roc_auc_score(y_test, stack_proba)
    stack_prauc = average_precision_score(y_test, stack_proba)
    print(f"  Augmented Stacking ROC-AUC: {stack_auc:.4f}")
    print(f"  Augmented Stacking PR-AUC:  {stack_prauc:.4f}")

    print("  Training simple meta-learner...")
    meta_simple = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
    meta_simple.fit(oof_preds, y_train)
    stack_simple_proba = meta_simple.predict_proba(test_stack_preds)[:, 1]
    stack_simple_auc = roc_auc_score(y_test, stack_simple_proba)
    stack_simple_prauc = average_precision_score(y_test, stack_simple_proba)
    print(f"  Simple Stacking ROC-AUC: {stack_simple_auc:.4f}")

    best_stack_proba = stack_proba if stack_auc > stack_simple_auc else stack_simple_proba
    best_stack_auc = max(stack_auc, stack_simple_auc)
    best_stack_prauc = stack_prauc if stack_auc > stack_simple_auc else stack_simple_prauc

    print("\n=== Evaluation ===")
    all_models = {
        "Logistic Regression": (lr_auc, lr_prauc, lr_proba),
        "XGBoost": (xgb_auc, xgb_prauc, xgb_proba),
        "LightGBM": (lgb_auc, lgb_prauc, lgb_proba),
        "CatBoost": (cb_auc, cb_prauc, cb_proba),
        "ExtraTrees": (et_auc, et_prauc, et_proba),
        "Stacking": (best_stack_auc, best_stack_prauc, best_stack_proba),
    }
    best_model_name = max(all_models, key=lambda k: all_models[k][0])
    best_auc, best_prauc, best_proba = all_models[best_model_name]
    print(f"  Best model: {best_model_name} (ROC-AUC={best_auc:.4f})")

    print("\n=== Multi-Seed Ensemble ===")
    seeds = [42, 123, 456, 789, 2024]
    xgb_seed_probas = []
    cb_seed_probas = []
    for seed in seeds:
        m = xgb.XGBClassifier(
            n_estimators=2000, **best_xgb_params,
            scale_pos_weight=pos_weight, random_state=seed,
            eval_metric="auc", early_stopping_rounds=80,
        )
        X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
        )
        m.fit(X_tr_s, y_tr_s, eval_set=[(X_val_s, y_val_s)], verbose=False)
        xgb_seed_probas.append(m.predict_proba(X_test)[:, 1])

        c = CatBoostClassifier(
            iterations=2000, **best_cb_params,
            auto_class_weights="Balanced", random_seed=seed,
            verbose=0, early_stopping_rounds=80,
        )
        c.fit(X_tr_s, y_tr_s, eval_set=(X_val_s, y_val_s), verbose=0)
        cb_seed_probas.append(c.predict_proba(X_test)[:, 1])
        print(f"  Seed {seed}: XGB={roc_auc_score(y_test, xgb_seed_probas[-1]):.4f}, CB={roc_auc_score(y_test, cb_seed_probas[-1]):.4f}")

    xgb_avg = np.mean(xgb_seed_probas, axis=0)
    cb_avg = np.mean(cb_seed_probas, axis=0)
    xgb_avg_auc = roc_auc_score(y_test, xgb_avg)
    cb_avg_auc = roc_auc_score(y_test, cb_avg)
    print(f"  XGB seed-avg ROC-AUC: {xgb_avg_auc:.4f}")
    print(f"  CB seed-avg ROC-AUC:  {cb_avg_auc:.4f}")

    mega_blend = 0.5 * xgb_avg + 0.5 * cb_avg
    mega_auc = roc_auc_score(y_test, mega_blend)
    mega_prauc = average_precision_score(y_test, mega_blend)
    print(f"  XGB+CB mega-blend ROC-AUC: {mega_auc:.4f}")
    print(f"  XGB+CB mega-blend PR-AUC:  {mega_prauc:.4f}")

    all_blend = 0.35 * xgb_avg + 0.35 * cb_avg + 0.15 * lr_proba + 0.15 * et_proba
    all_blend_auc = roc_auc_score(y_test, all_blend)
    all_blend_prauc = average_precision_score(y_test, all_blend)
    print(f"  Full blend ROC-AUC: {all_blend_auc:.4f}")
    print(f"  Full blend PR-AUC:  {all_blend_prauc:.4f}")

    final_best_auc = max(best_auc, mega_auc, all_blend_auc, xgb_avg_auc, cb_avg_auc)
    if mega_auc == final_best_auc:
        best_proba = mega_blend
        best_model_name = "XGB+CB Blend"
    elif all_blend_auc == final_best_auc:
        best_proba = all_blend
        best_model_name = "Full Blend"
    elif xgb_avg_auc == final_best_auc:
        best_proba = xgb_avg
        best_model_name = "XGB Seed-Avg"
    elif cb_avg_auc == final_best_auc:
        best_proba = cb_avg
        best_model_name = "CB Seed-Avg"
    best_auc = final_best_auc
    print(f"\n  FINAL BEST: {best_model_name} ROC-AUC = {best_auc:.4f}")

    all_models["XGB+CB Blend"] = (mega_auc, mega_prauc, mega_blend)
    all_models["Full Blend"] = (all_blend_auc, all_blend_prauc, all_blend)

    precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_thresh_idx = np.argmax(f1s)
    best_threshold = thresholds[best_thresh_idx] if best_thresh_idx < len(thresholds) else 0.5
    y_pred = (best_proba >= best_threshold).astype(int)

    perf_rows = []
    for name, (auc_val, prauc_val, _) in all_models.items():
        row = {"Model": name, "ROC-AUC": auc_val, "PR-AUC": prauc_val}
        if name == best_model_name:
            row["Precision"] = precision_score(y_test, y_pred)
            row["Recall"] = recall_score(y_test, y_pred)
            row["F1"] = f1_score(y_test, y_pred)
            row["Threshold"] = best_threshold
        perf_rows.append(row)
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(TABLES_DIR / "model_performance.csv", index=False)

    print("\n=== Cross-Validated Performance (Repeated 3x5) ===")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    cv_xgb = xgb.XGBClassifier(
        n_estimators=xgb_model.best_iteration,
        **best_xgb_params,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_val_score(cv_xgb, X, y, cv=rskf, scoring="roc_auc")
    print(f"  XGBoost 3x5-Fold CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("\n=== Generating plots ===")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (_, _, proba_val) in all_models.items():
        fpr, tpr, _ = roc_curve(y_test, proba_val)
        auc_val = roc_auc_score(y_test, proba_val)
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
    for name, (_, _, proba_val) in all_models.items():
        prec, rec, _ = precision_recall_curve(y_test, proba_val)
        ap = average_precision_score(y_test, proba_val)
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
