# %% [markdown]
# # Gas Concentration Regressor Training with Model Selection
# 
# This notebook implements **Stage 2** of the gas classification project with an **AutoML-style model selection**.
# 
# ## Objective
# Data-driven selection of the best regression algorithms for each gas type to predict concentration.
#
# ## Methodology
# 1. **Data Loading**: Load sensor dataset.
# 2. **Preprocessing**:
#    - Filter per gas.
#    - Target: Log1p transformation.
# 3. **Model Selection**:
#    Compare the following algorithms:
#    - **HistGradientBoostingRegressor (HGBR)**: Fast, gradient boosting, handles NaNs.
#    - **RandomForestRegressor (RF)**: Robust ensemble method, reduced overfitting.
#    - **ExtraTreesRegressor (ET)**: Extremely randomized trees, often faster/better than RF.
#    - **Ridge Regression**: Linear baseline with regularization (requires scaling).
# 4. **Evaluation**:
#    - Split Data (80/20).
#    - Train each candidate.
#    - Select best based on **RMSE** (Root Mean Squared Error).
# 5. **Persistence**: Save best models to `models/regressors/`.

# %% [code]
import sys
import os
from pathlib import Path
import time

# Add project root to path if running interactively
current_dir = Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from src.utils import REGRESSORS_DIR, logger
from src.data_loader import load_data, preprocess_for_regression
from src.classifier import VOC_LABELS

# %% [code]
# Load Dataset
df = load_data()

# %% [markdown]
# ## Candidate Definitions
# We define a dictionary of model pipelines.
# Note: Tree-based models (RF, ET, HGBR) generally don't need scaling, but Ridge does.

# %% [code]
def get_candidates():
    return {
        "HGBR": HistGradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])
    }

# %% [markdown]
# ## Training & Selection Loop

# %% [code]
results_summary = []

for gas_id, gas_name in VOC_LABELS.items():
    logger.info(f"\n{'='*30}\nProcessing {gas_name}\n{'='*30}")
    
    # 1. Data Prep
    X, y_log = preprocess_for_regression(df, gas_name)
    
    if X is None or len(X) < 50:
        logger.warning(f"Skipping {gas_name}: Insufficient data.")
        continue
        
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # Transform targets for metric calculation
    y_test_orig = np.expm1(y_test_log)
    
    best_loss = float('inf')
    best_model_name = None
    best_model_obj = None
    best_metrics = {}
    
    candidates = get_candidates()
    
    for name, model in candidates.items():
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train_log)
        
        # Predict
        y_pred_log = model.predict(X_test)
        
        # Safe inverse transform
        # Clip log predictions to avoid overflow (exp(100) is huge enough)
        y_pred_log = np.clip(y_pred_log, -2, 100)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0) # Clip negative predictions
        
        # Metrics
        mae = mean_absolute_error(y_test_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        r2 = r2_score(y_test_orig, y_pred)
        
        elapsed = time.time() - start_time
        logger.info(f"[{name}] RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f} | Time: {elapsed:.2f}s")
        
        # Selection Logic (Minimize RMSE)
        if rmse < best_loss:
            best_loss = rmse
            best_model_name = name
            best_model_obj = model
            best_metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
    logger.info(f"🏆 Winner for {gas_name}: {best_model_name} (RMSE: {best_loss:.3f})")
    
    # Save Winner
    save_path = REGRESSORS_DIR / f"{gas_name.lower()}_reg.pkl"
    joblib.dump(best_model_obj, save_path)
    
    results_summary.append({
        'Gas': gas_name,
        'Best Model': best_model_name,
        **best_metrics
    })

# %% [code]
# Final Report
summary_df = pd.DataFrame(results_summary)
print("\n=== Model Selection Summary ===")
print(summary_df)
summary_df.to_csv("regressor_selection_summary.csv", index=False)
