import pandas as pd
import numpy as np
from src.utils import DATA_FILE, logger

def load_data(filepath=DATA_FILE):
    """
    Load gas sensor data from CSV.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    logger.info(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows.")
    return df

def preprocess_for_regression(df, target_gas_name):
    """
    Filter data for a specific gas and prepare X, y.
    Applies log1p transformation to the target concentration.
    """
    # Filter by gas name
    gas_df = df[df['gas_name'] == target_gas_name].copy()
    
    if gas_df.empty:
        logger.warning(f"No data found for gas: {target_gas_name}")
        return None, None
    
    # Feature columns (Feature1 ... Feature128)
    # Assuming columns are named 'Feature1', 'Feature2', etc.
    # We can detect them dynamically or assume standard names based on notebook exploration
    feature_cols = [col for col in gas_df.columns if col.startswith("Feature")]
    
    X = gas_df[feature_cols]
    y = gas_df['concentration(ppmv)'] # Ensure this matches column name from notebook output
    
    # Handle NaNs if any (simple fill or drop)
    # Notebook showed X = X.fillna(X.mean())
    X = X.fillna(X.mean())
    
    # Log transform target (optional but recommended for concentration)
    # Plan mentioned: Apply log1p(y) transformation
    y_log = np.log1p(y)
    
    return X, y_log
