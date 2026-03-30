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
DEBUG = False

# ─── Feature Engineering ─────────────────────────────────
LOW_VARIANCE_THRESHOLD = 0.0
CORRELATION_THRESHOLD = 0.95

# ─── Models to run ───────────────────────────────────────
MODELS_TO_RUN = ["random_forest", "logistic", "xgboost"]