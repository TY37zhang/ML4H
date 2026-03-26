import pandas as pd

DRUG_CLASS_MAP = {
    "statin": ["SIMVASTATIN", "ATORVASTATIN", "PRAVASTATIN", "ROSUVASTATIN",
               "LOVASTATIN", "FLUVASTATIN", "PITAVASTATIN"],
    "ace_inhibitor": ["LISINOPRIL", "ENALAPRIL", "RAMIPRIL", "BENAZEPRIL",
                      "CAPTOPRIL", "FOSINOPRIL", "QUINAPRIL", "PERINDOPRIL"],
    "beta_blocker": ["METOPROLOL", "ATENOLOL", "CARVEDILOL", "PROPRANOLOL",
                     "BISOPROLOL", "NEBIVOLOL", "SOTALOL", "NADOLOL", "LABETALOL"],
    "thiazide": ["HYDROCHLOROTHIAZIDE", "CHLORTHALIDONE", "INDAPAMIDE", "METOLAZONE"],
    "ppi": ["OMEPRAZOLE", "ESOMEPRAZOLE", "PANTOPRAZOLE", "LANSOPRAZOLE",
            "RABEPRAZOLE", "DEXLANSOPRAZOLE"],
    "ccb": ["AMLODIPINE", "DILTIAZEM", "NIFEDIPINE", "VERAPAMIL", "FELODIPINE"],
    "metformin": ["METFORMIN"],
    "arb": ["LOSARTAN", "VALSARTAN", "IRBESARTAN", "CANDESARTAN",
            "TELMISARTAN", "OLMESARTAN", "AZILSARTAN"],
    "ssri": ["SERTRALINE", "CITALOPRAM", "FLUOXETINE", "ESCITALOPRAM", "PAROXETINE"],
    "opioid": ["HYDROCODONE", "OXYCODONE", "MORPHINE", "FENTANYL", "CODEINE",
               "TRAMADOL", "METHADONE", "HYDROMORPHONE", "BUPRENORPHINE"],
    "nsaid": ["IBUPROFEN", "NAPROXEN", "MELOXICAM", "CELECOXIB",
              "DICLOFENAC", "INDOMETHACIN", "PIROXICAM"],
    "antiepileptic": ["GABAPENTIN", "PREGABALIN", "TOPIRAMATE", "LAMOTRIGINE",
                      "LEVETIRACETAM", "VALPROIC", "DIVALPROEX", "CARBAMAZEPINE",
                      "PHENYTOIN", "ZONISAMIDE"],
    "loop_diuretic": ["FUROSEMIDE", "BUMETANIDE", "TORSEMIDE"],
    "alpha_blocker": ["TAMSULOSIN", "DOXAZOSIN", "TERAZOSIN", "PRAZOSIN",
                      "ALFUZOSIN", "SILODOSIN"],
    "gout_drug": ["ALLOPURINOL", "COLCHICINE", "FEBUXOSTAT", "PROBENECID"],
    "benzodiazepine": ["ALPRAZOLAM", "CLONAZEPAM", "LORAZEPAM", "DIAZEPAM",
                       "TEMAZEPAM", "MIDAZOLAM"],
    "sulfonylurea": ["GLIPIZIDE", "GLIMEPIRIDE", "GLYBURIDE"],
    "thyroid": ["LEVOTHYROXINE"],
    "bisphosphonate": ["ALENDRONATE", "RISEDRONATE", "IBANDRONATE", "ZOLEDRONIC"],
    "potassium_sparing": ["SPIRONOLACTONE", "TRIAMTERENE", "AMILORIDE", "EPLERENONE"],
}


def classify_drug(drug_name):
    drug_upper = str(drug_name).upper().strip()
    classes = []
    for cls_name, keywords in DRUG_CLASS_MAP.items():
        if any(kw in drug_upper for kw in keywords):
            classes.append(cls_name)
    return classes


def create_drug_class_matrix(rx_df):
    df = rx_df.copy()
    df["RXDDRUG"] = df["RXDDRUG"].astype(str).str.upper().str.strip()
    df = df[df["RXDDRUG"] != "NAN"]
    df = df[df["RXDDRUG"] != ""]

    if "RXDUSE" in df.columns:
        active = df[df["RXDUSE"] == 1.0]
    else:
        active = df

    person_drugs = active.groupby("SEQN")["RXDDRUG"].apply(list).reset_index()

    class_cols = {cls: [] for cls in DRUG_CLASS_MAP}
    seqns = []
    drug_counts = []

    for _, row in person_drugs.iterrows():
        seqns.append(row["SEQN"])
        drugs = row["RXDDRUG"]
        drug_counts.append(len(set(drugs)))
        person_classes = set()
        for d in drugs:
            person_classes.update(classify_drug(d))
        for cls in DRUG_CLASS_MAP:
            class_cols[cls].append(1 if cls in person_classes else 0)

    result = pd.DataFrame({"SEQN": seqns, "drug_count": drug_counts})
    for cls in DRUG_CLASS_MAP:
        result[f"drug_{cls}"] = class_cols[cls]

    return result


def create_individual_drug_matrix(rx_df, top_n=50):
    df = rx_df.copy()
    df["RXDDRUG"] = df["RXDDRUG"].astype(str).str.upper().str.strip()
    df = df[(df["RXDDRUG"] != "NAN") & (df["RXDDRUG"] != "")]

    drug_counts = df.groupby("RXDDRUG")["SEQN"].nunique().sort_values(ascending=False)
    top_drugs = drug_counts.head(top_n).index.tolist()

    person_drugs = df.drop_duplicates(subset=["SEQN", "RXDDRUG"])
    person_drugs = person_drugs[person_drugs["RXDDRUG"].isin(top_drugs)]

    pivot = person_drugs.pivot_table(index="SEQN", columns="RXDDRUG", aggfunc="size", fill_value=0)
    pivot = (pivot > 0).astype(int)
    pivot.columns = [f"ind_drug_{c.lower().replace(' ', '_').replace(';', '_')}" for c in pivot.columns]

    return pivot.reset_index()
