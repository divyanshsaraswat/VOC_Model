import joblib
import numpy as np
import pandas as pd
from src.utils import CLASSIFIER_MODEL, SCALER_FILE, LABEL_ENCODER_FILE, logger

# Helper mapping matching app.py
VOC_LABELS = {
    0: "Ethanol",
    1: "Ethylene",
    2: "Ammonia",
    3: "Acetaldehyde",
    4: "Acetone",
    5: "Toluene"
}

class GasClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._load_assets()

    def _load_assets(self):
        if not CLASSIFIER_MODEL.exists():
            raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_MODEL}")
        
        logger.info("Loading classifier assets...")
        self.model = joblib.load(CLASSIFIER_MODEL)
        
        if SCALER_FILE.exists():
            self.scaler = joblib.load(SCALER_FILE)
        else:
            logger.warning("Scaler not found, raw features will be used.")

        if LABEL_ENCODER_FILE.exists():
             self.label_encoder = joblib.load(LABEL_ENCODER_FILE)

    def predict(self, feature_data):
        """
        Predict gas category from features.
        Args:
            feature_data (list or np.array or pd.DataFrame): 128 features.
        Returns:
            str: Predicted gas name
        """
        # Ensure 2D array
        if isinstance(feature_data, list):
            feature_data = np.array(feature_data).reshape(1, -1)
        elif isinstance(feature_data, pd.DataFrame):
             feature_data = feature_data.values
        elif isinstance(feature_data, pd.Series):
             feature_data = feature_data.values.reshape(1, -1)
        
        if feature_data.ndim == 1:
            feature_data = feature_data.reshape(1, -1)

        # Scale
        if self.scaler:
            X_scaled = self.scaler.transform(feature_data)
        else:
            X_scaled = feature_data
            
        # Predict
        # Model returns numeric label
        pred_label_idx = self.model.predict(X_scaled)[0]
        
        # Decode label
        # App.py uses VOC_LABELS mapping (0->Ethanol, etc) which is reliable for this model.
        # The label_encoder.pkl seems to map to gas_id (int), which causes issues.
        # We will use VOC_LABELS for string names.
        gas_name = VOC_LABELS.get(int(pred_label_idx), "Unknown")
            
        return gas_name
