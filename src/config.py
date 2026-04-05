import os

# ─── Paths ────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
INTERIM_DATA_PATH   = os.path.join(BASE_DIR, "data", "interim")

ARTIFACTS_PATH  = os.path.join(BASE_DIR, "artifacts")
MODELS_PATH     = os.path.join(BASE_DIR, "artifacts", "model")
SCALERS_PATH    = os.path.join(BASE_DIR, "artifacts", "scalers")
REPORTS_PATH    = os.path.join(BASE_DIR, "artifacts", "reports")
LOGS_PATH       = os.path.join(BASE_DIR, "logs")

# ─── Dataset ──────────────────────────────────────────────
# All CICIDS CSV files (in the order you want them processed)
RAW_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

PROCESSED_DATA_VERSION = "v1"
PROCESSED_FILENAME = f"processed_data_{PROCESSED_DATA_VERSION}.csv"

# ─── Columns ──────────────────────────────────────────────
TARGET_COLUMN = "Label"  
COLUMNS_TO_DROP = ["Flow ID", " Source IP", " Destination IP",
                   " Timestamp", "Fwd Header Length.1"]

# ─── Preprocessing ────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
SCALE_FEATURES = True

# ─── Model ────────────────────────────────────────────────
MODEL_TYPE = "random_forest"   # options: "random_forest", "xgboost", "logistic"

# ─── Flags ────────────────────────────────────────────────
SAVE_PROCESSED_DATA = True
SAVE_MODEL          = True
SAVE_SCALER         = True

# ─── Debug ────────────────────────────────────────────────
DEBUG = True

# ─── Feature Engineering ─────────────────────────────────
LOW_VARIANCE_THRESHOLD = 0.0
CORRELATION_THRESHOLD = 0.95

# ── Baseline training — models to train in main.py ───────────────────────────
# SVM is placed last — it is the slowest on 2.8M records (~5-10 min).
MODELS_TO_RUN = [
    "random_forest",
    "xgboost",
    "logistic",
    "decision_tree",
    "svm",
]

# ── Cross-dataset evaluation — models to use in testdrift.py ─────────────────
# All five models are compared for drift sensitivity.
# SVM uses scaled data (already handled in cross_dataset_evaluation).
CROSS_EVAL_MODELS = [
    "random_forest",
    "xgboost",
    "logistic",
    "decision_tree",
    "svm",
]

# ── Drift metrics ─────────────────────────────────────────────────────────────
# Toggle which divergence metrics are computed in compute_drift().
COMPUTE_KL          = True
COMPUTE_JS          = True
COMPUTE_WASSERSTEIN = True   # Wasserstein distance (Earth Mover's Distance)

# ── Rolling reference window ──────────────────────────────────────────────────
# If True, compute_rolling_drift() compares each batch against the
# immediately preceding batch instead of all batches against Monday.
USE_ROLLING_REFERENCE = True

# ── Drift alert ───────────────────────────────────────────────────────────────
# If the drift predictor's predicted accuracy falls below this threshold,
# a drift alert is written to artifacts/reports/drift_alerts.json.
DRIFT_ALERT_THRESHOLD = 0.80

# ── SHAP ──────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE    = 500    # rows sampled from drifted batch for SHAP
SHAP_TOP_N_FEATURES = 15     # number of top features to show in SHAP plot
