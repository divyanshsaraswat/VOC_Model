import joblib
import numpy as np
from src.utils import REGRESSORS_DIR, logger

class ConcentrationRegressor:
    def __init__(self, gas_name):
        self.gas_name = gas_name
        self.model_path = REGRESSORS_DIR / f"{gas_name.lower()}_reg.pkl"
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"No regressor found for {self.gas_name} at {self.model_path}")
            return
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded regressor for {self.gas_name}")
        except Exception as e:
            logger.error(f"Failed to load regressor for {self.gas_name}: {e}")

    def predict(self, feature_data):
        """
        Predict concentration.
        Args:
            feature_data: scaled or raw features? 
            Note: The regressors will be trained on RAW features or SCALED features?
            Usually regressors like Trees (HistGradientBoosting) don't strictly need scaling, 
            but consistency is good. 
            However, the classifier used scaling. 
            Let's decide: The training script will determine this. 
            For now, we accept `feature_data` and assume the caller provides correct format.
        """
        if not self.model:
            return None
        
        try:
            # Predict log1p target
            pred_log = self.model.predict(feature_data)[0]
            
            # Inverse log1p -> expm1
            concentration = np.expm1(pred_log)
            
            # Ensure non-negative
            return max(0.0, float(concentration))
        except Exception as e:
            logger.error(f"Prediction error for {self.gas_name}: {e}")
            return None
