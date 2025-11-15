"""
Streamlit ML Prediction Application
Production-ready application for loading and running predictions with .pkl models
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="ML Model Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path="model.pkl"):
    """
    Load the ML model with caching for performance.
    Uses @st.cache_resource to ensure model loads only once.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        
        model = joblib.load(model_path)
        
        # Check if model is very large and warn
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        if model_size > 100:
            st.warning(f"⚠️ Model file is large ({model_size:.2f} MB). This may slow down loading.")
        
        return model, None
    except Exception as e:
        return None, str(e)


def validate_inputs(input_data, expected_features=None):
    """
    Validate user inputs before prediction.
    """
    errors = []
    
    if expected_features:
        if len(input_data) != expected_features:
            errors.append(f"Expected {expected_features} features, got {len(input_data)}")
    
    # Check for NaN or None values
    if any(pd.isna(x) or x is None for x in input_data):
        errors.append("Some input fields contain invalid values")
    
    # Check for numeric types
    for i, val in enumerate(input_data):
        try:
            float(val)
        except (ValueError, TypeError):
            errors.append(f"Feature {i+1} must be a number")
    
    return errors


def get_model_info(model):
    """
    Extract model information for display.
    """
    info = {
        "type": type(model).__name__,
        "has_predict_proba": hasattr(model, "predict_proba"),
        "n_features": None,
        "classes": None
    }
    
    # Try to get feature count from model
    if hasattr(model, "n_features_in_"):
        info["n_features"] = model.n_features_in_
    elif hasattr(model, "n_features_"):
        info["n_features"] = model.n_features_
    elif hasattr(model, "coef_") and model.coef_ is not None:
        if len(model.coef_.shape) > 0:
            info["n_features"] = model.coef_.shape[-1]
    
    # Try to get classes for classification
    if hasattr(model, "classes_"):
        info["classes"] = model.classes_.tolist()
    
    return info


def create_default_inputs(n_features=4):
    """
    Create default input fields based on number of features.
    This is a generic implementation - customize based on your model.
    """
    inputs = []
    
    with st.container():
        cols = st.columns(min(3, n_features))
        
        for i in range(n_features):
            with cols[i % len(cols)]:
                # You can customize input types here
                # For now, using number inputs
                value = st.number_input(
                    f"Feature {i+1}",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.1,
                    key=f"feature_{i}",
                    help=f"Enter value for feature {i+1}"
                )
                inputs.append(value)
    
    return inputs


# Sidebar
with st.sidebar:
    st.markdown("## 📋 About This Model")
    
    st.markdown("""
    ### Model Information
    This application loads a pre-trained machine learning model
    and allows you to make predictions interactively.
    
    ### Version
    **v1.0.0**
    
    ### Dataset Description
    The model was trained on a custom dataset with features
    that need to be provided as inputs.
    
    ### How to Use
    1. Enter values for all required features
    2. Click the "Predict" button
    3. View the prediction and confidence scores
    4. Check the history section for previous predictions
    
    ### GitHub Repository
    [View on GitHub](https://github.com/yourusername/voc-streamlit)
    
    ### Instructions
    - Ensure all fields are filled
    - Use valid numeric values
    - Check error messages if prediction fails
    """)
    
    # Dark mode toggle (optional)
    dark_mode = st.checkbox("🌙 Dark Mode (Coming Soon)", value=False)


# Main Content
st.markdown('<div class="main-header">🤖 ML Model Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Load Model
model, error = load_model()

if error:
    st.markdown(f'<div class="error-box"><h3>❌ Error Loading Model</h3><p>{error}</p></div>', unsafe_allow_html=True)
    st.stop()

if model is None:
    st.markdown('<div class="error-box"><h3>❌ Model Not Found</h3><p>Please ensure model.pkl exists in the project directory.</p></div>', unsafe_allow_html=True)
    st.stop()

# Get model information
model_info = get_model_info(model)

# Display Model Information
with st.expander("ℹ️ Model Details", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", model_info["type"])
    with col2:
        st.metric("Supports Probabilities", "Yes" if model_info["has_predict_proba"] else "No")
    with col3:
        if model_info["n_features"]:
            st.metric("Number of Features", model_info["n_features"])
    
    if model_info["classes"]:
        st.write("**Classes:**", ", ".join(map(str, model_info["classes"])))

st.markdown("---")

# Input Section
st.markdown("## 📥 Input Features")

# Initialize session state for prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Get number of features (default to 4 if not detected)
n_features = model_info["n_features"] or 4

# Create input fields
st.markdown("### Enter Feature Values")

# For production, you should customize these inputs based on your actual model
# This is a generic implementation
feature_inputs = create_default_inputs(n_features)

# Display input summary
with st.expander("📊 Input Summary"):
    input_df = pd.DataFrame({
        "Feature": [f"Feature {i+1}" for i in range(len(feature_inputs))],
        "Value": feature_inputs
    })
    st.dataframe(input_df, use_container_width=True)

st.markdown("---")

# Prediction Section
st.markdown("## 🎯 Prediction")

col1, col2 = st.columns([1, 4])

with col1:
    predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

if predict_button:
    # Validate inputs
    validation_errors = validate_inputs(feature_inputs, n_features)
    
    if validation_errors:
        error_msg = "\n".join([f"• {err}" for err in validation_errors])
        st.markdown(f'<div class="error-box"><h3>❌ Validation Error</h3><p>{error_msg}</p></div>', unsafe_allow_html=True)
    else:
        try:
            # Convert inputs to numpy array
            input_array = np.array(feature_inputs).reshape(1, -1)
            
            # Make prediction
            with st.spinner("🔄 Running prediction..."):
                prediction = model.predict(input_array)[0]
                
                # Get probabilities if available
                proba = None
                if model_info["has_predict_proba"]:
                    proba = model.predict_proba(input_array)[0]
            
            # Display Results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### ✅ Prediction Result")
            
            # Format prediction based on type
            if isinstance(prediction, (np.integer, int)):
                pred_display = int(prediction)
            elif isinstance(prediction, (np.floating, float)):
                pred_display = f"{prediction:.4f}"
            else:
                pred_display = str(prediction)
            
            st.markdown(f"**Predicted Value:** `{pred_display}`")
            
            # Display probabilities if available
            if proba is not None:
                st.markdown('<div class="confidence-box">', unsafe_allow_html=True)
                st.markdown("### 📊 Confidence Scores")
                
                # Create probability dataframe
                if model_info["classes"]:
                    prob_df = pd.DataFrame({
                        "Class": model_info["classes"],
                        "Probability": proba
                    }).sort_values("Probability", ascending=False)
                else:
                    prob_df = pd.DataFrame({
                        "Class": [f"Class {i}" for i in range(len(proba))],
                        "Probability": proba
                    }).sort_values("Probability", ascending=False)
                
                st.dataframe(prob_df, use_container_width=True)
                
                # Visualize probabilities
                st.bar_chart(prob_df.set_index("Class")["Probability"])
                
                # Show highest probability
                max_prob_idx = prob_df["Probability"].idxmax()
                max_class = prob_df.loc[max_prob_idx, "Class"]
                max_prob = prob_df.loc[max_prob_idx, "Probability"]
                st.metric("Highest Confidence", f"{max_class}: {max_prob:.2%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save to history
            history_entry = {
                "inputs": feature_inputs.copy(),
                "prediction": pred_display,
                "probabilities": proba.tolist() if proba is not None else None
            }
            st.session_state.prediction_history.insert(0, history_entry)
            
            # Keep only last 10 predictions
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history = st.session_state.prediction_history[:10]
            
        except Exception as e:
            st.markdown(f'<div class="error-box"><h3>❌ Prediction Error</h3><p>{str(e)}</p><p><small>Please check your inputs and try again.</small></p></div>', unsafe_allow_html=True)

# Prediction History
if st.session_state.prediction_history:
    st.markdown("---")
    st.markdown("## 📜 Prediction History")
    
    history_data = []
    for idx, entry in enumerate(st.session_state.prediction_history[:5]):  # Show last 5
        history_data.append({
            "Index": idx + 1,
            "Inputs": str(entry["inputs"])[:50] + "..." if len(str(entry["inputs"])) > 50 else str(entry["inputs"]),
            "Prediction": entry["prediction"]
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with ❤️ using Streamlit</p>
    <p>Production-Ready ML Prediction App v1.0.0</p>
</div>
""", unsafe_allow_html=True)

