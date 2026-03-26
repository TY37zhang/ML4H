import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from config import *
from drug_classes import create_drug_class_matrix

def load_dataset(prefix, columns=None):
    frames = []
    for suffix, info in CYCLES.items():
        path = RAW_DIR / f"{prefix}_{suffix}.xpt"
        if not path.exists():
            continue
        try:
            df = pd.read_sas(path, format="xport", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_sas(path, format="xport", encoding="latin-1")
        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[["SEQN"] + [c for c in available if c != "SEQN"]]
        df["cycle"] = suffix
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_egfr(row):
    scr = row["LBXSCR"]
    age = row["RIDAGEYR"]
    sex = row["RIAGENDR"]
    if pd.isna(scr) or pd.isna(age) or pd.isna(sex):
        return np.nan
    if sex == 2:
        kappa, alpha, sex_mult = 0.7, -0.241, 1.012
    else:
        kappa, alpha, sex_mult = 0.9, -0.302, 1.0
    scr_ratio = scr / kappa
    egfr = 142 * (min(scr_ratio, 1.0) ** alpha) * (max(scr_ratio, 1.0) ** -1.200) * (0.9938 ** age) * sex_mult
    return egfr


def main():
    print("=== Loading datasets ===")

    kiq_cols = ["SEQN", "KIQ026", "KID028"]
    kiq = load_dataset("KIQ_U", kiq_cols)
    print(f"  KIQ_U: {len(kiq)} rows")

    demo_cols = ["SEQN"] + list(DEMO_VARS.keys()) + ["WTMEC2YR"]
    demo = load_dataset("DEMO", demo_cols)
    print(f"  DEMO: {len(demo)} rows")

    bio_cols = ["SEQN"] + list(LAB_VARS.keys())
    bio = load_dataset("BIOPRO", bio_cols)
    print(f"  BIOPRO: {len(bio)} rows")

    urine_cols = ["SEQN"] + list(URINE_VARS.keys())
    urine = load_dataset("ALB_CR", urine_cols)
    print(f"  ALB_CR: {len(urine)} rows")

    body_cols = ["SEQN"] + list(BODY_VARS.keys())
    body = load_dataset("BMX", body_cols)
    print(f"  BMX: {len(body)} rows")

    bp_cols = ["SEQN", "BPXSY1", "BPXDI1", "BPXOSY1", "BPXODI1"]
    bp = load_dataset("BPX", bp_cols)
    for sys_col, dia_col in [("BPXSY1", "BPXDI1"), ("BPXOSY1", "BPXODI1")]:
        if sys_col in bp.columns and "systolic_bp" not in bp.columns:
            bp["systolic_bp"] = bp[sys_col]
            bp["diastolic_bp"] = bp[dia_col]
        elif sys_col in bp.columns:
            bp["systolic_bp"] = bp["systolic_bp"].fillna(bp[sys_col])
            bp["diastolic_bp"] = bp["diastolic_bp"].fillna(bp[dia_col])
    bp = bp[["SEQN", "cycle", "systolic_bp", "diastolic_bp"]].copy()
    print(f"  BPX: {len(bp)} rows")

    mcq_cols = ["SEQN", "MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ220"]
    mcq = load_dataset("MCQ", mcq_cols)
    print(f"  MCQ: {len(mcq)} rows")

    diq_cols = ["SEQN", "DIQ010"]
    diq = load_dataset("DIQ", diq_cols)
    print(f"  DIQ: {len(diq)} rows")

    bpq_cols = ["SEQN", "BPQ020", "BPQ080"]
    bpq = load_dataset("BPQ", bpq_cols)
    print(f"  BPQ: {len(bpq)} rows")

    smq_cols = ["SEQN", "SMQ020", "SMQ040"]
    smq = load_dataset("SMQ", smq_cols)
    print(f"  SMQ: {len(smq)} rows")

    slq_cols = ["SEQN", "SLD012"]
    slq = load_dataset("SLQ", slq_cols)
    print(f"  SLQ: {len(slq)} rows")

    paq_cols = ["SEQN", "PAQ605", "PAQ650"]
    paq = load_dataset("PAQ", paq_cols)
    print(f"  PAQ: {len(paq)} rows")

    rx_cols = ["SEQN", "RXDUSE", "RXDDRUG", "RXDDRGID", "RXDDAYS", "RXDCOUNT"]
    rx = load_dataset("RXQ_RX", rx_cols)
    print(f"  RXQ_RX: {len(rx)} rows")

    print("\n=== Creating drug class matrix ===")
    drug_matrix = create_drug_class_matrix(rx)
    print(f"  Drug matrix: {len(drug_matrix)} persons, {len(drug_matrix.columns)} columns")

    print("\n=== Merging datasets by cycle ===")
    merged_frames = []
    for suffix in CYCLES:
        k = kiq[kiq["cycle"] == suffix].drop(columns=["cycle"])
        d = demo[demo["cycle"] == suffix].drop(columns=["cycle"])
        b = bio[bio["cycle"] == suffix].drop(columns=["cycle"])
        u = urine[urine["cycle"] == suffix].drop(columns=["cycle"])
        bo = body[body["cycle"] == suffix].drop(columns=["cycle"])
        bpx = bp[bp["cycle"] == suffix].drop(columns=["cycle"])
        m = mcq[mcq["cycle"] == suffix].drop(columns=["cycle"])
        di = diq[diq["cycle"] == suffix].drop(columns=["cycle"])
        bq = bpq[bpq["cycle"] == suffix].drop(columns=["cycle"])
        sm = smq[smq["cycle"] == suffix].drop(columns=["cycle"])
        sl = slq[slq["cycle"] == suffix].drop(columns=["cycle"])
        pa = paq[paq["cycle"] == suffix].drop(columns=["cycle"])

        merged = k.merge(d, on="SEQN", how="left")
        for right_df in [b, u, bo, bpx, m, di, bq, sm, sl, pa]:
            merged = merged.merge(right_df, on="SEQN", how="left")

        merged["cycle"] = suffix
        merged_frames.append(merged)
        print(f"  Cycle {suffix}: {len(merged)} rows")

    df = pd.concat(merged_frames, ignore_index=True)
    print(f"\n  Pooled: {len(df)} rows")

    df = df.merge(drug_matrix, on="SEQN", how="left")
    drug_cols = [c for c in drug_matrix.columns if c.startswith("drug_") or c == "drug_count"]
    df[drug_cols] = df[drug_cols].fillna(0)

    print("\n=== Feature engineering ===")

    df = df[~df["KIQ026"].isin(OUTCOME_EXCLUDE)]
    df = df[df["KIQ026"].isin([OUTCOME_YES, OUTCOME_NO])]
    df["kidney_stones"] = (df["KIQ026"] == OUTCOME_YES).astype(int)
    print(f"  After outcome filter: {len(df)} rows, {df['kidney_stones'].sum()} stone cases")

    n_cycles = len(CYCLES)
    df["survey_weight"] = df["WTMEC2YR"] / n_cycles

    df["egfr"] = df.apply(compute_egfr, axis=1)

    df["acr"] = np.where(
        (df["URXUCR"].notna()) & (df["URXUCR"] > 0),
        df["URXUMA"] / df["URXUCR"] * 100,
        np.nan
    )

    df["smoking_status"] = np.where(
        df["SMQ020"] == 2, 0,
        np.where(
            (df["SMQ020"] == 1) & (df["SMQ040"].isin([1, 2])), 2,
            np.where(
                (df["SMQ020"] == 1) & (df["SMQ040"] == 3), 1, np.nan
            )
        )
    )

    df["diabetes_status"] = np.where(
        df["DIQ010"] == 1, 1,
        np.where(df["DIQ010"] == 2, 0,
                 np.where(df["DIQ010"] == 3, 2, np.nan))
    )

    df["physically_active"] = np.where(
        (df["PAQ605"] == 1) | (df["PAQ650"] == 1), 1,
        np.where(
            (df["PAQ605"] == 2) & (df["PAQ650"] == 2), 0, np.nan
        )
    )

    df["obesity_category"] = pd.cut(
        df["BMXBMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3],
        right=False
    ).astype(float)

    df["polypharmacy"] = (df["drug_count"] >= 5).astype(int)

    for var in COMORBIDITY_VARS:
        if var in df.columns:
            df[var] = np.where(df[var] == 1, 1, np.where(df[var] == 2, 0, np.nan))

    df["sex_binary"] = (df["RIAGENDR"] == 1).astype(int)

    print("\n=== Handling missing data ===")
    feature_cols = (
        list(LAB_VARS.keys()) + list(URINE_VARS.keys()) + list(BODY_VARS.keys()) +
        ["systolic_bp", "diastolic_bp", "egfr", "acr", "SLD012",
         "smoking_status", "diabetes_status", "physically_active", "obesity_category",
         "RIDAGEYR", "INDFMPIR", "DMDEDUC2"] +
        list(COMORBIDITY_VARS.keys())
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    missingness = {}
    for col in feature_cols:
        n_miss = df[col].isna().sum()
        pct = n_miss / len(df) * 100
        missingness[col] = {"n_missing": n_miss, "pct_missing": round(pct, 1)}
        if 0 < pct < 50:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    miss_df = pd.DataFrame(missingness).T
    miss_df.to_csv(TABLES_DIR / "missingness.csv")
    print(f"  Missingness table saved")

    exclude_cols = [col for col, info in missingness.items() if info["pct_missing"] >= 50]
    if exclude_cols:
        print(f"  Excluding (>50% missing): {exclude_cols}")

    numeric_cols = [c for c in feature_cols if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, float]]
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    print(f"\n=== Final dataset ===")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Kidney stones: {df['kidney_stones'].sum()} ({df['kidney_stones'].mean()*100:.1f}%)")
    print(f"  Cycles: {df['cycle'].value_counts().to_dict()}")

    output_path = PROCESSED_DIR / "analysis_ready.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
