import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
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

    diet_cols = ["SEQN"] + list(DIETARY_VARS.keys())
    diet = load_dataset("DR1TOT", diet_cols)
    print(f"  DR1TOT: {len(diet)} rows")

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
        dt = diet[diet["cycle"] == suffix].drop(columns=["cycle"])

        merged = k.merge(d, on="SEQN", how="left")
        for right_df in [b, u, bo, bpx, m, di, bq, sm, sl, pa, dt]:
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

    df["ca_phos_ratio"] = np.where(
        (df["LBXSPH"].notna()) & (df["LBXSPH"] > 0),
        df["LBXSCA"] / df["LBXSPH"], np.nan
    )
    df["bun_cr_ratio"] = np.where(
        (df["LBXSCR"].notna()) & (df["LBXSCR"] > 0),
        df["LBXSBU"] / df["LBXSCR"], np.nan
    )
    df["na_k_ratio"] = np.where(
        (df["LBXSKSI"].notna()) & (df["LBXSKSI"] > 0),
        df["LBXSNASI"] / df["LBXSKSI"], np.nan
    )
    df["bun_egfr_ratio"] = np.where(
        (df["egfr"].notna()) & (df["egfr"] > 0),
        df["LBXSBU"] / df["egfr"], np.nan
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

    for val, name in [(1.0, "mexican_american"), (2.0, "other_hispanic"),
                      (3.0, "nh_white"), (4.0, "nh_black"), (5.0, "other_race")]:
        df[f"race_{name}"] = (df["RIDRETH1"] == val).astype(int)

    df["age_squared"] = df["RIDAGEYR"] ** 2
    df["bmi_squared"] = df["BMXBMI"] ** 2
    df["age_x_sex"] = df["RIDAGEYR"] * df["sex_binary"]
    df["age_x_bmi"] = df["RIDAGEYR"] * df["BMXBMI"]
    df["log_drug_count"] = np.log1p(df["drug_count"])

    df["age_group_young"] = ((df["RIDAGEYR"] >= 20) & (df["RIDAGEYR"] < 40)).astype(int)
    df["age_group_middle"] = ((df["RIDAGEYR"] >= 40) & (df["RIDAGEYR"] < 60)).astype(int)
    df["age_group_senior"] = (df["RIDAGEYR"] >= 60).astype(int)

    df["drug_count_x_age"] = df["drug_count"] * df["RIDAGEYR"]
    df["drug_count_x_bmi"] = df["drug_count"] * df["BMXBMI"]
    df["drug_count_x_egfr"] = df["drug_count"] * df["egfr"]
    df["uric_acid_x_bmi"] = df["LBXSUA"] * df["BMXBMI"]
    df["calcium_x_age"] = df["LBXSCA"] * df["RIDAGEYR"]
    df["egfr_x_age"] = df["egfr"] * df["RIDAGEYR"]
    df["waist_x_sex"] = df["BMXWAIST"] * df["sex_binary"]
    df["bp_product"] = df["systolic_bp"] * df["diastolic_bp"]
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["anion_gap"] = df["LBXSNASI"] - df["LBXSCLSI"] - df["LBXSC3SI"]
    df["ca_cr_product"] = df["LBXSCA"] * df["LBXSCR"]

    high_risk_drugs = ["drug_gout_drug", "drug_beta_blocker", "drug_opioid",
                       "drug_thiazide", "drug_ppi"]
    df["high_risk_drug_count"] = sum(
        df[d].fillna(0) for d in high_risk_drugs if d in df.columns
    )
    df["any_diuretic"] = ((df.get("drug_loop_diuretic", 0) == 1) |
                          (df.get("drug_thiazide", 0) == 1) |
                          (df.get("drug_potassium_sparing", 0) == 1)).astype(int)
    df["metabolic_syndrome_score"] = (
        (df["BMXBMI"] >= 30).astype(int) +
        df["diabetes_status"].fillna(0).clip(0, 1).astype(int) +
        df.get("BPQ020", pd.Series(0, index=df.index)).fillna(0).astype(int) +
        (df["LBXSGL"] >= 100).astype(int)
    )

    df["sodium_potassium_ratio"] = np.where(
        (df["DR1TPOTA"].notna()) & (df["DR1TPOTA"] > 0),
        df["DR1TSODI"] / df["DR1TPOTA"], np.nan
    )
    df["calcium_per_kg"] = np.where(
        (df["BMXBMI"].notna()) & (df["BMXBMI"] > 0),
        df["DR1TCALC"] / df["BMXBMI"], np.nan
    )
    df["low_water_intake"] = (df["DR1TMOIS"] < 1500).astype(int)
    df["high_sodium_intake"] = (df["DR1TSODI"] > 2300).astype(int)
    df["high_protein_intake"] = (df["DR1TPROT"] > 100).astype(int)

    print("\n=== Handling missing data ===")
    feature_cols = (
        list(LAB_VARS.keys()) + list(URINE_VARS.keys()) + list(BODY_VARS.keys()) +
        ["systolic_bp", "diastolic_bp", "egfr", "acr", "SLD012",
         "smoking_status", "diabetes_status", "physically_active", "obesity_category",
         "RIDAGEYR", "INDFMPIR", "DMDEDUC2",
         "ca_phos_ratio", "bun_cr_ratio", "na_k_ratio", "bun_egfr_ratio",
         "drug_count_x_age", "drug_count_x_bmi", "drug_count_x_egfr",
         "uric_acid_x_bmi", "calcium_x_age", "egfr_x_age",
         "waist_x_sex", "bp_product", "pulse_pressure", "anion_gap", "ca_cr_product",
         "sodium_potassium_ratio", "calcium_per_kg"] +
        list(DIETARY_VARS.keys()) +
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

    lab_cols_to_mice = [c for c in numeric_cols if c.startswith("LBX") or
                        c in ["egfr", "acr", "ca_phos_ratio", "bun_cr_ratio",
                              "na_k_ratio", "bun_egfr_ratio"]]
    other_cols_to_impute = [c for c in numeric_cols if c not in lab_cols_to_mice]

    if lab_cols_to_mice:
        print(f"  MICE imputation on {len(lab_cols_to_mice)} lab columns...")
        mice_imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=False)
        df[lab_cols_to_mice] = mice_imputer.fit_transform(df[lab_cols_to_mice])
    if other_cols_to_impute:
        simple_imputer = SimpleImputer(strategy="median")
        df[other_cols_to_impute] = simple_imputer.fit_transform(df[other_cols_to_impute])

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
