from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
MODELS_DIR = PROJECT_DIR / "outputs" / "models"
NHANES_CACHE = Path("/Users/tianyinzhang/DEV/nhanes_cache")

RANDOM_STATE = 42

PRESCRIPTION_RECALL_DAYS = 30
LONG_TERM_USE_DAYS = 365
TWO_YEAR_USE_DAYS = 730
INVALID_DURATION_CODES = [77777, 99999]

ESTIMAND_SPEC = {
    "time_zero": "NHANES household interview / prescription medication inventory date",
    "primary_exposure": "Any reported use of a prescription drug class in the past 30 days at time zero",
    "duration_sensitivity": "Current drug class use with RXDDAYS >= 365 or >= 730 days",
    "outcome": "Lifetime self-reported kidney stone history as of time zero (KIQ026)",
    "follow_up": "No prospective follow-up is observed in NHANES; estimates are interpreted as adjusted prevalence effects/associations under a target-trial-style framework",
}

CYCLES = {
    "E": {"year": 2007, "label": "2007-2008"},
    "F": {"year": 2009, "label": "2009-2010"},
    "G": {"year": 2011, "label": "2011-2012"},
    "H": {"year": 2013, "label": "2013-2014"},
    "I": {"year": 2015, "label": "2015-2016"},
    "J": {"year": 2017, "label": "2017-2018"},
}

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{filename}.xpt"

CACHED_DATASETS = ["KIQ_U", "RXQ_RX", "BIOPRO", "DEMO", "ALB_CR"]
DOWNLOAD_DATASETS = ["MCQ", "DIQ", "BPQ", "SMQ", "BMX", "BPX", "SLQ", "PAQ", "DR1TOT"]

OUTCOME_VAR = "KIQ026"
OUTCOME_YES = 1.0
OUTCOME_NO = 2.0
OUTCOME_EXCLUDE = [7.0, 9.0]

DEMO_VARS = {
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "RIDRETH1": "race_ethnicity",
    "INDFMPIR": "poverty_income_ratio",
    "DMDEDUC2": "education",
}

LAB_VARS = {
    "LBXSCA": "serum_calcium",
    "LBXSUA": "serum_uric_acid",
    "LBXSPH": "serum_phosphorus",
    "LBXSC3SI": "serum_bicarbonate",
    "LBXSBU": "serum_bun",
    "LBXSCR": "serum_creatinine",
    "LBXSNASI": "serum_sodium",
    "LBXSKSI": "serum_potassium",
    "LBXSCLSI": "serum_chloride",
    "LBXSGL": "serum_glucose",
    "LBXSCH": "serum_cholesterol",
}

URINE_VARS = {
    "URXUMA": "urine_albumin",
    "URXUCR": "urine_creatinine",
}

BODY_VARS = {
    "BMXBMI": "bmi",
    "BMXWAIST": "waist_circumference",
}

BP_EXAM_VARS = {
    "BPXSY1": "systolic_bp",
    "BPXDI1": "diastolic_bp",
}

COMORBIDITY_VARS = {
    "MCQ160B": "chf",
    "MCQ160C": "chd",
    "MCQ160E": "heart_attack",
    "MCQ160F": "stroke",
    "MCQ220": "cancer",
    "DIQ010": "diabetes",
    "BPQ020": "hypertension",
    "BPQ080": "high_cholesterol",
}

LIFESTYLE_VARS = {
    "SMQ020": "smoked_100_cigs",
    "SMQ040": "current_smoking",
    "SLD012": "sleep_hours",
    "PAQ605": "vigorous_activity",
    "PAQ650": "moderate_activity",
}

DIETARY_VARS = {
    "DR1TKCAL": "energy_kcal",
    "DR1TPROT": "protein_g",
    "DR1TSODI": "sodium_mg",
    "DR1TPOTA": "potassium_mg",
    "DR1TCALC": "calcium_mg",
    "DR1TMAGN": "magnesium_mg",
    "DR1TVC": "vitamin_c_mg",
    "DR1TMOIS": "moisture_g",
    "DR1TSUGR": "total_sugars_g",
    "DR1TFIBE": "fiber_g",
}

DRUG_MIN_USERS = 100
TOP_INDIVIDUAL_DRUGS = 50
CONFOUNDING_EXCLUSION_DRUGS = ["TAMSULOSIN"]
