"""
Configuration for the KOA multi-backbone HHO-ensemble pipeline.
Values match the final settings reported in the manuscript (Table 3).

Dataset paths are intentionally left as placeholders: the OAI radiograph
set and Clinical Data-2 tabular set are not distributed in this
repository (imaging data governed by the OAI data use agreement;
clinical data is confidential patient data, see the paper's Ethical
Statement). Point DATA_ROOT at your own local OAI export before running.
"""
import os

DATA_ROOT = os.environ.get("KOA_DATA_ROOT", "./data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

NUM_CLASSES = 5  # KL grades 0-4
IMAGE_SIZE = 224
RANDOM_SEED = 42

BACKBONES = ["densenet121", "efficientnetv2_s", "resnet50", "convnext_tiny"]

# --- Phase 1: backbone fine-tuning ---
HEAD_LR_DEFAULT = 1e-4
HEAD_LR_CONVNEXT = 5e-5
BACKBONE_LR_RATIO = 0.1
WEIGHT_DECAY_DEFAULT = 1e-4
WEIGHT_DECAY_CONVNEXT = 0.05
MAX_EPOCHS_DEFAULT = 20
MAX_EPOCHS_CONVNEXT = 30
EARLY_STOP_PATIENCE = 5
CONVNEXT_WARMUP_EPOCHS = 5
LLRD_DECAY = 0.65
LLRD_GROUPS = 6

# Combined loss: (1 - lambda_f) * CORN + lambda_f * BSFL
LAMBDA_F = 0.4
BSFL_GAMMA = 2
KL1_WEIGHT_MULTIPLIER = 4.0
OVERSAMPLE_BETA = 0.7  # sampling weight ~ n_c^(-beta)

# --- Phase 2: ensemble weight optimization ---
ENSEMBLE_POPULATION = 30
ENSEMBLE_ITERATIONS = 80

# --- Test-time augmentation ---
TTA_VARIANTS = 8

# --- Clinical branch (HHO feature selection + SVM) ---
CLINICAL_HHO_POPULATION = 15
CLINICAL_HHO_ITERATIONS = 30
CLINICAL_KL_GRADES = (2, 3, 4)  # KL 0/1 not present in this cohort
