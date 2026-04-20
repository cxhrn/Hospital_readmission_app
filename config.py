from pathlib import Path

PROJECT_ROOT = Path.cwd()
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

DATA_PATH = PROJECT_ROOT / "diabetic_data.csv"

RANDOM_STATE = 42
TARGET_COL = "readmitted_binary"

NUMERICAL_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

DROP_COLS = [
    "readmitted",
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty",
]

DEATH_CODES = [11, 13, 14, 19, 20, 21]

APP_COLUMNS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
    "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin",
    "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed"
]
