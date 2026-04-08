"""Block 1 configuration: paths, field mappings, hyperparameters."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("HEARTAGE_DATA_DIR",
                               PROJECT_ROOT / "data" / "UKB"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "block1"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# --- 27 CMR IDPs organized by 5 domains ---
# Latent dims: LV 3, RV 2, Atrial 2, Aortic 2, Mechanics 3 → total 12
IDP_DOMAINS = {
    "LV": {
        "fields": [
            "p24100_i2",  # LV end diastolic volume (mL)
            "p24101_i2",  # LV end systolic volume (mL)
            "p24102_i2",  # LV stroke volume (mL)
            "p24103_i2",  # LV ejection fraction (%)
            "p24104_i2",  # LV cardiac output (L/min)
            "p24105_i2",  # LV myocardial mass (g)
        ],
        "latent_dim": 3,
    },
    "RV": {
        "fields": [
            "p24106_i2",  # RV end diastolic volume (mL)
            "p24107_i2",  # RV end systolic volume (mL)
            "p24108_i2",  # RV stroke volume (mL)
            "p24109_i2",  # RV ejection fraction (%)
        ],
        "latent_dim": 2,
    },
    "Atrial": {
        "fields": [
            "p24110_i2",  # LA maximum volume (mL)
            "p24111_i2",  # LA minimum volume (mL)
            "p24112_i2",  # LA stroke volume (mL)
            "p24113_i2",  # LA ejection fraction (%)
            "p24114_i2",  # RA maximum volume (mL)
            "p24115_i2",  # RA minimum volume (mL)
            "p24116_i2",  # RA stroke volume (mL)
            "p24117_i2",  # RA ejection fraction (%)
        ],
        "latent_dim": 2,
    },
    "Aortic": {
        "fields": [
            "p24118_i2",  # Ascending aorta max area (mm²)
            "p24119_i2",  # Ascending aorta min area (mm²)
            "p24120_i2",  # Ascending aorta distensibility
            "p24121_i2",  # Descending aorta max area (mm²)
            "p24122_i2",  # Descending aorta min area (mm²)
            "p24123_i2",  # Descending aorta distensibility
        ],
        "latent_dim": 2,
    },
    "Mechanics": {
        "fields": [
            "p24157_i2",  # LV circumferential strain global (%)
            "p24174_i2",  # LV radial strain global (%)
            "p24181_i2",  # LV longitudinal strain global (%)
        ],
        "latent_dim": 3,
    },
}

# Flatten all IDP field names
ALL_IDP_FIELDS = []
DOMAIN_NAMES = []
DOMAIN_SLICES = {}  # domain_name -> (start_idx, end_idx) in ALL_IDP_FIELDS
_offset = 0
for dname, dinfo in IDP_DOMAINS.items():
    n = len(dinfo["fields"])
    ALL_IDP_FIELDS.extend(dinfo["fields"])
    DOMAIN_NAMES.append(dname)
    DOMAIN_SLICES[dname] = (_offset, _offset + n)
    _offset += n

TOTAL_IDP_DIM = len(ALL_IDP_FIELDS)  # 27
TOTAL_LATENT_DIM = sum(d["latent_dim"] for d in IDP_DOMAINS.values())  # 12

# --- Demographic / covariate fields ---
AGE_FIELD = "p21003_i2"   # Age at imaging visit
SEX_FIELD = "p31"          # Sex (0=Female, 1=Male)
BMI_FIELD = "p21001_i2"   # BMI
SMOKING_FIELD = "p20116_i2"  # Smoking status (0=Never, 1=Previous, 2=Current)
ACTIVITY_FIELD = "p904_i2"   # Days/week vigorous activity

# --- Source RDS files (map logical name → file) ---
RDS_SOURCES = {
    "heart_mri":   DATA_DIR / "heart_mri.rds",
    "recruitment": DATA_DIR / "Recruitment.rds",
    "population":  DATA_DIR / "Population_characteristics.rds",
    "hesin_diag":  DATA_DIR / "hesin_diag.rds",
    "physical":    DATA_DIR / "Physical_measures.rds",
    "lifestyle":   DATA_DIR / "Lifestyle_and_environment.rds",
}

# --- ICD-10 exclusion codes for strict_healthy ---
EXCLUSION_ICD10_PREFIXES = [
    "I50",                          # Heart failure
    "I42", "I43",                   # Cardiomyopathy
    "I21", "I22",                   # Myocardial infarction
    "I48",                          # Atrial fibrillation / flutter
    "I05", "I06", "I07", "I08", "I09",  # Rheumatic valvular
    "I34", "I35", "I36", "I37",         # Non-rheumatic valvular
    "I10", "I11", "I12", "I13", "I14", "I15",  # Hypertension
    "E10", "E11", "E12", "E13", "E14",  # Diabetes mellitus
    "N183", "N184", "N185",         # CKD stage 3–5
]
EXCLUSION_ICD10_EXACT = ["I252"]    # Old MI

# --- B-spline normative basis ---
SPLINE_DEGREE = 3
SPLINE_N_INTERIOR_KNOTS = 8

# --- Training ---
TRAIN_CFG = {
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 0.01,   # = L_reg
    "max_epochs": 300,
    "patience": 30,
    "seed": 42,
    # Loss weights
    "w_norm": 1.0,
    "w_age": 0.5,
    "w_rank": 0.10,
    # Ablation only
    "w_rec": 0.1,
}

# --- Split ratios ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Model architecture ---
ENCODER_HIDDEN_MIN = 8
MONO_HIDDEN = [32, 16]
BASELINE_B_HIDDEN = [64, 32]
