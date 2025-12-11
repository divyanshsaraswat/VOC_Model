import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classifier import GasClassifier
from src.regressors import ConcentrationRegressor
from src.utils import logger

def predict_pipeline(input_data):
    """
    Run full pipeline on input data.
    input_data: list or 1D array of 128 features
    """
    # 1. Classify
    clf = GasClassifier()
    gas_name = clf.predict(input_data)
    
    logger.info(f"Stage 1 Classification: {gas_name}")
    
    if gas_name == "Unknown":
        return {"gas": "Unknown", "concentration": None}
    
    # 2. Regress
    reg = ConcentrationRegressor(gas_name)
    if not reg.model:
        return {"gas": gas_name, "concentration": None, "error": "Regressor not found"}
    
    # Regressor expects 2D array?
    # Our data_loader returns X which is DataFrame.
    # classifier.predict accepts 1D/2D.
    # regressor.predict should handle it.
    
    # Note: Regressor might need reshaped input
    if isinstance(input_data, list):
        input_data = np.array(input_data).reshape(1, -1)
    
    concentration = reg.predict(input_data)
    
    return {
        "gas": gas_name,
        "concentration": concentration
    }

def main():
    parser = argparse.ArgumentParser(description="Gas Classification and Concentration Prediction")
    parser.add_argument("--input", type=str, required=True, help="Path to JSON file containing features (list of 128 floats)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(json.dumps({"error": f"File not found: {input_path}"}))
        sys.exit(1)
        
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        # Expecting data['features'] or just a list?
        # Let's support both list or {"features": [...]}
        if isinstance(data, dict) and "features" in data:
            features = data["features"]
        elif isinstance(data, list):
            features = data
        else:
            raise ValueError("Invalid JSON format. Expected list of features or object with 'features' key.")
            
        if len(features) != 128:
             # Just a warning or strict? 
             # Classifier trains on 128.
             # We assume input is correct size.
             pass

        result = predict_pipeline(features)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.exception("Prediction failed")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
