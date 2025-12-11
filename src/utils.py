import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
REGRESSORS_DIR = MODELS_DIR / "regressors"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGRESSORS_DIR.mkdir(parents=True, exist_ok=True)

# Files
# Use exact filenames from your environment
DATA_FILE = DATA_DIR / "gas_sensor_data.csv"
CLASSIFIER_MODEL = MODELS_DIR / "classifier.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl" 
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoding.pkl"

def check_structure():
    """Verify critical paths exist."""
    paths = [DATA_DIR, MODELS_DIR, REGRESSORS_DIR]
    for p in paths:
        if not p.exists():
            logger.warning(f"Directory not found: {p}")
            # p.mkdir(parents=True, exist_ok=True) # Optional auto-create

    if not DATA_FILE.exists():
         logger.error(f"Data file missing at {DATA_FILE}")

