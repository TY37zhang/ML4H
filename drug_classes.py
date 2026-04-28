import pandas as pd
from config import INVALID_DURATION_CODES, LONG_TERM_USE_DAYS, TWO_YEAR_USE_DAYS

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


def clean_duration_days(duration):
    duration = pd.to_numeric(duration, errors="coerce")
    duration = duration.mask(duration.isin(INVALID_DURATION_CODES) | (duration < 0))
    return duration


def create_drug_class_matrix(rx_df):
    df = rx_df.copy()
    df["RXDDRUG"] = df["RXDDRUG"].astype(str).str.upper().str.strip()
    df = df[df["RXDDRUG"] != "NAN"]
    df = df[df["RXDDRUG"] != ""]

    if "RXDUSE" in df.columns:
        active = df[df["RXDUSE"] == 1.0]
    else:
        active = df

    active = active.copy()
    if "RXDDAYS" in active.columns:
        active["rx_days_clean"] = clean_duration_days(active["RXDDAYS"])
    else:
        active["rx_days_clean"] = pd.NA

    records = []
    for seqn, person in active.groupby("SEQN"):
        drugs = person["RXDDRUG"].tolist()
        person_classes = set()
        class_days = {cls: [] for cls in DRUG_CLASS_MAP}

        for drug_name, duration in zip(person["RXDDRUG"], person["rx_days_clean"]):
            for cls in classify_drug(drug_name):
                person_classes.add(cls)
                if pd.notna(duration):
                    class_days[cls].append(duration)

        record = {"SEQN": seqn, "drug_count": len(set(drugs))}
        for cls in DRUG_CLASS_MAP:
            max_days = max(class_days[cls]) if class_days[cls] else 0
            record[f"drug_{cls}"] = 1 if cls in person_classes else 0
            record[f"drug_{cls}_days_max"] = max_days
            record[f"drug_{cls}_long_term"] = int(max_days >= LONG_TERM_USE_DAYS)
            record[f"drug_{cls}_two_year"] = int(max_days >= TWO_YEAR_USE_DAYS)
        records.append(record)

    return pd.DataFrame(records)


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
