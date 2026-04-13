"""Block 2 configuration: ECG student distillation from CMR teacher."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("HEARTAGE_DATA_DIR",
                               PROJECT_ROOT / "data" / "UKB"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "block2"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Block 1 outputs (teacher checkpoint + predictions)
BLOCK1_DIR = PROJECT_ROOT / "results" / "block1"
TEACHER_CKPT = BLOCK1_DIR / "models" / "full_teacher.pt"
TEACHER_PRED = BLOCK1_DIR / "predictions" / "full_teacher_pred.csv"
COHORT_STATS = BLOCK1_DIR / "cohort_stats.json"
SPLINE_CONFIG = BLOCK1_DIR / "spline_config.json"

# External ECG datasets
PTBXL_DIR = PROJECT_ROOT / "data" / "PTB-XL"
PTBXL_PLUS_DIR = PROJECT_ROOT / "data" / "PTB-XL+"
CODE15_DIR = PROJECT_ROOT / "data" / "CODE-15%"
MIMIC_ECG_DIR = PROJECT_ROOT / "data" / "MIMIC-IV-ECG"

# --- ECG preprocessing ---
ECG_SAMPLE_RATE = 500        # Hz (target)
ECG_DURATION_SEC = 10        # seconds
ECG_N_LEADS = 12
ECG_SEQ_LEN = ECG_SAMPLE_RATE * ECG_DURATION_SEC  # 5000 samples
ECG_BANDPASS_LOW = 0.5       # Hz
ECG_BANDPASS_HIGH = 40.0     # Hz

# Standard 12-lead order
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# --- Concept Bottleneck Ontology (12 concepts) ---
# Concepts 0-4: chronic structural (supervised by teacher domain_scores)
# Concepts 5-8: rhythm/morphology (supervised by PTB-XL labels)
# Concepts 9-11: quality/state (supervised by PTB-XL quality + rule-based QC)
CONCEPT_NAMES = [
    # Chronic structural (from teacher)
    "LV_deviation",           # 0
    "RV_deviation",           # 1
    "Atrial_deviation",       # 2
    "Aortic_deviation",       # 3
    "Mechanics_deviation",    # 4
    # Rhythm / morphology (from PTB-XL)
    "sinus_rhythm_conf",      # 5
    "QRS_duration_norm",      # 6
    "ST_deviation_score",     # 7
    "P_wave_quality",         # 8
    # Quality / state (from PTB-XL + rule-based)
    "signal_noise_ratio",     # 9
    "baseline_stability",     # 10
    "lead_completeness",      # 11
]
N_CONCEPTS = len(CONCEPT_NAMES)
STRUCTURAL_CONCEPT_IDX = list(range(5))   # concepts 0-4
RHYTHM_CONCEPT_IDX = list(range(5, 9))    # concepts 5-8
QUALITY_CONCEPT_IDX = list(range(9, 12))  # concepts 9-11

# Teacher domain names → concept index mapping
DOMAIN_TO_CONCEPT = {
    "LV": 0, "RV": 1, "Atrial": 2, "Aortic": 3, "Mechanics": 4,
}

# PTB-XL SCP code groups for rhythm concept supervision
PTBXL_RHYTHM_LABELS = {
    "sinus_rhythm_conf": ["SR", "NORM"],       # sinus rhythm / normal
    "QRS_duration_norm": ["LBBB", "RBBB", "LAFB", "LPFB", "WPW"],  # conduction
    "ST_deviation_score": ["STD_", "STE_", "ISCA", "ISCAL", "ISCI"],  # ST changes
    "P_wave_quality": ["AFIB", "AFLT", "PAC", "SVTAC"],  # P-wave / atrial
}

# --- ECG Encoder Architecture ---
# ResNet-1D backbone
ENCODER_CFG = {
    "in_channels": ECG_N_LEADS,       # 12 leads
    "base_filters": 64,
    "n_blocks": 4,                    # 4 residual blocks
    "kernel_size": 15,
    "stride": 2,
    "dropout": 0.1,
    "embedding_dim": 256,             # output embedding size
}

# --- Student Model Architecture ---
STUDENT_CFG = {
    "embedding_dim": ENCODER_CFG["embedding_dim"],
    "n_concepts": N_CONCEPTS,
    "concept_hidden": 128,
    "main_head_hidden": [64, 32],
    "mc_dropout_rate": 0.1,           # for scope_uncertainty
    "mc_samples": 20,                 # MC dropout forward passes
}

# --- Training ---
TRAIN_CFG = {
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "max_epochs": 100,
    "patience": 15,
    "seed": 42,
    # Loss weights
    "w_main": 1.0,          # MSE structural_age distillation
    "w_rank": 0.1,          # ranking loss (age ordering)
    "w_concept_struct": 0.5, # structural concept supervision (teacher domain scores)
    "w_concept_rhythm": 0.3, # rhythm concept supervision (PTB-XL labels)
    "w_concept_quality": 0.2, # quality concept supervision
    # Encoder fine-tuning
    "encoder_lr_mult": 0.1,  # lower LR for pretrained encoder
    "warmup_epochs": 5,
}

# --- Derived score weights (not trained, fixed) ---
PERTURBATION_WEIGHTS = {
    "w_sinus": 1.0,       # (1 - sinus_rhythm_conf)
    "w_qrs": 0.5,         # |QRS_duration_norm - expected|
    "w_st": 0.8,          # ST_deviation_score
    "w_snr": 0.3,         # (1 - signal_noise_ratio)
}

SCOPE_WEIGHTS = {
    "alpha_mc": 0.5,       # MC dropout variance
    "beta_mahal": 0.3,     # Mahalanobis distance
    "gamma_miss": 0.2,     # missingness / quality penalty
}

# --- Split ratios (must match Block 1 for paired cohort) ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
