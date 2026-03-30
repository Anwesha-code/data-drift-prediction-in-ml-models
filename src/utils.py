import os
import logging
from datetime import datetime
from config import LOGS_PATH, REPORTS_PATH

def create_all_folders():
    """Create all required project directories if they don't exist."""
    from config import (PROCESSED_DATA_PATH, INTERIM_DATA_PATH,
                        MODELS_PATH, SCALERS_PATH, REPORTS_PATH, LOGS_PATH)
    folders = [PROCESSED_DATA_PATH, INTERIM_DATA_PATH,
               MODELS_PATH, SCALERS_PATH, REPORTS_PATH, LOGS_PATH]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger that writes to both console and a log file."""
    os.makedirs(LOGS_PATH, exist_ok=True)
    log_file = os.path.join(LOGS_PATH, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

def save_experiment_report(report: dict, filename: str = None):
    """Save experiment results as a text file in artifacts/reports/."""
    os.makedirs(REPORTS_PATH, exist_ok=True)
    if filename is None:
        filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    filepath = os.path.join(REPORTS_PATH, filename)
    with open(filepath, "w") as f:
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
    print(f"Report saved → {filepath}")