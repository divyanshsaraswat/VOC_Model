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
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

# Direct VOC label mapping (0-based indexing)
VOC_LABELS = {
    0: "Ethanol",
    1: "Ethylene",
    2: "Ammonia",
    3: "Acetaldehyde",
    4: "Acetone",
    5: "Toluene"
}

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
        color: #000000;
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
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-card h2 {
        font-size: 3rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .prediction-card p {
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        background-color: #4caf50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .confidence-medium {
        background-color: #ff9800;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .confidence-low {
        background-color: #f44336;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .safe-badge {
        background-color: #4caf50;
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .hazardous-badge {
        background-color: #f44336;
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
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


@st.cache_resource
def load_scaler(scaler_path="scaler.pkl"):
    """
    Load the scaler with caching for performance.
    Uses @st.cache_resource to ensure scaler loads only once.
    """
    try:
        if not os.path.exists(scaler_path):
            return None, f"Scaler file '{scaler_path}' not found. Predictions will run without scaling."
        
        scaler = joblib.load(scaler_path)
        return scaler, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_label_encoder(encoder_path="label_encoding.pkl"):
    """
    Load the label encoder with caching for performance.
    Uses @st.cache_resource to ensure label encoder loads only once.
    """
    try:
        if not os.path.exists(encoder_path):
            return None, f"Label encoder file '{encoder_path}' not found. Predictions will use numeric labels."
        
        label_encoder = joblib.load(encoder_path)
        return label_encoder, None
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
        "classes": None,
        "has_feature_importances": hasattr(model, "feature_importances_"),
        "has_coef": hasattr(model, "coef_") and model.coef_ is not None
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


def get_confidence_level(probability):
    """
    Determine confidence level based on probability.
    """
    if probability >= 0.8:
        return "High", "confidence-high"
    elif probability >= 0.6:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"


def get_human_readable_label(prediction, model_info, label_encoder=None):
    """
    Convert numeric prediction to human-readable label.
    Direct mapping: 1: Ethanol, 2: Ethylene, 3: Ammonia, 4: Acetaldehyde, 5: Acetone, 6: Toluene
    """
    
    # Convert prediction to int if numeric
    if isinstance(prediction, (int, np.integer)):
        pred_int = int(prediction)
        return VOC_LABELS.get(pred_int, str(prediction))
    elif isinstance(prediction, (float, np.floating)):
        pred_int = int(round(prediction))
        return VOC_LABELS.get(pred_int, str(prediction))
    elif isinstance(prediction, np.ndarray):
        pred_int = int(prediction[0]) if len(prediction) > 0 else 0
        return VOC_LABELS.get(pred_int, str(prediction))
    
    # Return raw prediction as string
    return str(prediction)


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

# Load Model, Scaler, and Label Encoder
model, model_error = load_model()
scaler, scaler_error = load_scaler()
label_encoder, label_encoder_error = load_label_encoder()

if model_error:
    st.markdown(f'<div class="error-box"><h3>❌ Error Loading Model</h3><p>{model_error}</p></div>', unsafe_allow_html=True)
    st.stop()

if model is None:
    st.markdown('<div class="error-box"><h3>❌ Model Not Found</h3><p>Please ensure model.pkl exists in the project directory.</p></div>', unsafe_allow_html=True)
    st.stop()

# Show model ready status
st.success("✅ Model is ready to go!")
    
# Show warnings for missing optional files (collapsed in expander)
if (scaler_error and scaler is None) or (label_encoder_error and label_encoder is None):
    with st.expander("ℹ️ Optional Files Status", expanded=False):
        if scaler_error and scaler is None:
            st.info(f"⚠️ {scaler_error}")
        if label_encoder_error and label_encoder is None:
            st.info(f"⚠️ {label_encoder_error}")

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

# Initialize session state for input method
if "input_method" not in st.session_state:
    st.session_state.input_method = "manual"
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_row_index" not in st.session_state:
    st.session_state.csv_row_index = 0

# Input Method Selection
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    input_method = st.radio(
        "Choose Input Method:",
        ["📝 Manual Input", "📄 Upload CSV File"],
        horizontal=True,
        key="input_method_radio"
    )

with col3:
    # Link button to download test CSV file
    st.link_button(
        label="📩 Download an example CSV File",
        url="https://ik.imagekit.io/mtuq5zmd4/test_data_1.csv?updatedAt=1763208492961",
        help="Download a sample test CSV file with example data",
        width='stretch'
    )

st.markdown("---")

# Handle CSV Upload
if input_method == "📄 Upload CSV File":
    st.markdown("### 📄 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with features. Each row represents one prediction set."
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            csv_df = pd.read_csv(uploaded_file)
            st.session_state.csv_data = csv_df
            
            # Validate CSV columns
            if len(csv_df.columns) != n_features:
                st.error(f"❌ CSV file must have exactly {n_features} columns (features). "
                        f"Your file has {len(csv_df.columns)} columns.")
                st.info(f"Expected columns: {', '.join([f'Feature_{i+1}' for i in range(n_features)])}")
                csv_df = None
            else:
                # Display CSV preview
                st.success(f"✅ CSV file loaded successfully! Found {len(csv_df)} row(s).")
                
                with st.expander("📊 CSV Preview", expanded=True):
                    st.dataframe(csv_df, width='stretch')
                
                # If multiple rows, let user select which row to use
                if len(csv_df) > 1:
                    st.markdown("**Multiple rows detected. Select which row to use for prediction:**")
                    selected_row = st.selectbox(
                        "Select Row",
                        range(len(csv_df)),
                        format_func=lambda x: f"Row {x+1}: {list(csv_df.iloc[x].values)}",
                        key="csv_row_selector"
                    )
                    st.session_state.csv_row_index = selected_row
                    feature_inputs = csv_df.iloc[selected_row].values.tolist()
                else:
                    st.session_state.csv_row_index = 0
                    feature_inputs = csv_df.iloc[0].values.tolist()
                    
                # Display selected row
                st.markdown("**Selected Row for Prediction:**")
                selected_row_df = pd.DataFrame({
                    "Feature": [f"Feature {i+1}" for i in range(len(feature_inputs))],
                    "Value": feature_inputs
                })
                st.dataframe(selected_row_df, width='stretch')
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with numeric values only.")
            feature_inputs = None
    else:
        st.info("ℹ️ Please upload a CSV file to use this input method.")
        feature_inputs = None

else:
    # Manual Input Mode
    st.markdown("### 📝 Enter Feature Values Manually")
    
    # For production, you should customize these inputs based on your actual model
    # This is a generic implementation
    feature_inputs = create_default_inputs(n_features)
    
    # Display input summary
    with st.expander("📊 Input Summary", expanded=False):
        input_df = pd.DataFrame({
            "Feature": [f"Feature {i+1}" for i in range(len(feature_inputs))],
            "Value": feature_inputs
        })
        st.dataframe(input_df, width='stretch')

# Store current inputs in session state
if feature_inputs is not None:
    st.session_state.current_inputs = feature_inputs
else:
    if "current_inputs" not in st.session_state:
        # Initialize with zeros if no valid inputs
        st.session_state.current_inputs = [0.0] * n_features
    feature_inputs = st.session_state.current_inputs

st.markdown("---")

# Create Tabs for Organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction", 
    "📊 Probability Analysis", 
    "📈 Feature Importance", 
    "📉 Input Visualization",
    "🔬 Sensor Analysis"
])

# Initialize session state for storing prediction results
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None
if "last_scaled_inputs" not in st.session_state:
    st.session_state.last_scaled_inputs = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None

# TAB 1: PREDICTION
with tab1:
    st.markdown("## 🎯 Prediction")
    
    # Check if CSV data is available with multiple rows
    has_csv_data = st.session_state.csv_data is not None and len(st.session_state.csv_data) > 1
    can_batch_predict = has_csv_data
    
    col1, col2 = st.columns([1, 1]) if can_batch_predict else st.columns([1, 4])
    
    with col1:
        predict_button = st.button("🔮 Predict", type="primary", width='stretch')
    
    # Batch prediction button if CSV has multiple rows
    batch_predict_button = None
    if can_batch_predict:
        with col2:
            batch_predict_button = st.button("📊 Batch Predict (All Rows)", type="secondary", width='stretch', key="batch_predict_btn")
    
    if predict_button:
        # Use current inputs from session state if available
        if "current_inputs" in st.session_state:
            feature_inputs = st.session_state.current_inputs
        
        # Validate inputs
        validation_errors = validate_inputs(feature_inputs, n_features)
        
        if validation_errors:
            error_msg = "\n".join([f"• {err}" for err in validation_errors])
            st.error(f"**Validation Error:**\n{error_msg}")
        else:
            try:
                # Convert inputs to numpy array
                input_array = np.array(feature_inputs).reshape(1, -1)
                
                # Scale inputs if scaler is available
                scaled_inputs = input_array
                if scaler is not None:
                    try:
                        scaled_inputs = scaler.transform(input_array)
                        st.session_state.last_scaled_inputs = scaled_inputs[0].tolist()
                    except Exception as e:
                        st.error(f"⚠️ Scaling failed: {str(e)}. Using unscaled inputs.")
                        scaled_inputs = input_array
                        st.session_state.last_scaled_inputs = None
                else:
                    st.session_state.last_scaled_inputs = None
                
                # Make prediction
                with st.spinner("🔄 Running prediction..."):
                    prediction = model.predict(scaled_inputs)[0]
                    
                    # Get probabilities if available
                    proba = None
                    if model_info["has_predict_proba"]:
                        proba = model.predict_proba(scaled_inputs)[0]
                
                # Store in session state
                st.session_state.last_prediction = prediction
                st.session_state.last_inputs = feature_inputs.copy()
                st.session_state.last_proba = proba.tolist() if proba is not None else None
                
                # Format prediction based on type
                if isinstance(prediction, (np.integer, int)):
                    pred_display = int(prediction)
                elif isinstance(prediction, (np.floating, float)):
                    pred_display = f"{prediction:.4f}"
                else:
                    pred_display = str(prediction)
                
                # Get human-readable label
                pred_label = get_human_readable_label(prediction, model_info, label_encoder)
                
                # Display Large Prediction Card
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f'<h2>{pred_label}</h2>', unsafe_allow_html=True)
                if isinstance(prediction, (np.integer, int)) and pred_label == str(prediction):
                    st.markdown(f'<p>Prediction: {pred_display}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display Confidence Score (for classification)
                if proba is not None:
                    # Use the prediction value directly as index for probabilities
                    if isinstance(prediction, (int, np.integer)):
                        pred_idx = int(prediction)
                        # Ensure index is within bounds
                        if pred_idx < len(proba):
                            confidence = proba[pred_idx]
                        else:
                            confidence = np.max(proba)
                            pred_idx = np.argmax(proba)
                    else:
                        # For non-integer predictions, use max probability
                        confidence = np.max(proba)
                        pred_idx = np.argmax(proba)
                    conf_level, conf_class = get_confidence_level(confidence)
                    
                    st.markdown(f'<div class="{conf_class}">Confidence: {conf_level} ({confidence:.2%})</div>', unsafe_allow_html=True)
                    
                    # Pass/Fail Indicator (if applicable - customize threshold)
                    if model_info["classes"]:
                        # Customize this based on your model
                        # Example: Class 0 = Safe, Class 1 = Hazardous
                        if len(model_info["classes"]) == 2 and confidence > 0.7:
                            if pred_idx == 0:  # Assuming 0 is safe
                                st.markdown('<div class="safe-badge">✅ SAFE</div>', unsafe_allow_html=True)
                            elif pred_idx == 1:  # Assuming 1 is hazardous
                                st.markdown('<div class="hazardous-badge">⚠️ HAZARDOUS</div>', unsafe_allow_html=True)
                
                # Save to history
                history_entry = {
                    "inputs": feature_inputs.copy(),
                    "prediction": pred_display,
                    "label": pred_label,
                    "probabilities": proba.tolist() if proba is not None else None
                }
                if "prediction_history" not in st.session_state:
                    st.session_state.prediction_history = []
                st.session_state.prediction_history.insert(0, history_entry)
                
                # Keep only last 10 predictions
                if len(st.session_state.prediction_history) > 10:
                    st.session_state.prediction_history = st.session_state.prediction_history[:10]
                
                st.success("✅ Prediction completed successfully!")
                
            except Exception as e:
                st.error(f"**Prediction Error:** {str(e)}\n\nPlease check your inputs and try again.")
    
    # Batch Prediction (if CSV has multiple rows)
    if batch_predict_button:
        if st.session_state.csv_data is not None:
            csv_df = st.session_state.csv_data
            st.markdown("---")
            st.markdown("### 📊 Batch Predictions")
            
            try:
                # Validate all rows
                valid_rows = []
                invalid_rows = []
                
                for idx, row in csv_df.iterrows():
                    row_values = row.values.tolist()
                    errors = validate_inputs(row_values, n_features)
                    if errors:
                        invalid_rows.append((idx + 1, errors))
                    else:
                        valid_rows.append((idx, row_values))
                
                if invalid_rows:
                    st.warning(f"⚠️ {len(invalid_rows)} row(s) have validation errors and will be skipped:")
                    for row_num, errors in invalid_rows:
                        st.write(f"Row {row_num}: {', '.join(errors)}")
                
                if valid_rows:
                    with st.spinner(f"🔄 Running batch predictions on {len(valid_rows)} row(s)..."):
                        predictions_list = []
                        probabilities_list = []
                        
                        for row_idx, row_values in valid_rows:
                            try:
                                # Convert to numpy array
                                input_array = np.array(row_values).reshape(1, -1)
                                
                                # Scale inputs if scaler is available
                                scaled_inputs = input_array
                                if scaler is not None:
                                    try:
                                        scaled_inputs = scaler.transform(input_array)
                                    except Exception:
                                        scaled_inputs = input_array
                                
                                # Make prediction
                                prediction = model.predict(scaled_inputs)[0]
                                
                                # Get probabilities if available
                                proba = None
                                if model_info["has_predict_proba"]:
                                    proba = model.predict_proba(scaled_inputs)[0]
                                
                                predictions_list.append({
                                    "Row": row_idx + 1,
                                    "Prediction": prediction,
                                    "Label": get_human_readable_label(prediction, model_info, label_encoder),
                                    "Probabilities": proba.tolist() if proba is not None else None
                                })
                            except Exception as e:
                                st.error(f"Error predicting row {row_idx + 1}: {str(e)}")
                        
                        # Create results dataframe (only predictions and probabilities, no input features)
                        if predictions_list:
                            # Create a mapping of row indices to predictions
                            pred_dict = {p["Row"] - 1: p for p in predictions_list}
                            
                            # Create new dataframe with only Row, Prediction, and Probabilities
                            results_data = {
                                "Row": [],
                                "Prediction": []
                            }
                            
                            # Add predictions (mapped using label_encoder)
                            for idx in range(len(csv_df)):
                                if idx in pred_dict:
                                    results_data["Row"].append(idx + 1)
                                    # Prediction already mapped using label_encoder in get_human_readable_label
                                    results_data["Prediction"].append(pred_dict[idx]["Label"])
                                else:
                                    # Skip invalid rows in results
                                    continue
                            
                            # Add probabilities if available
                            if predictions_list and predictions_list[0]["Probabilities"] is not None:
                                # Use direct VOC labels mapping (defined at top of file)
                                # Determine class names based on number of probabilities
                                num_classes = len(predictions_list[0]["Probabilities"])
                                class_names = [VOC_LABELS.get(i, f"Class_{i}") for i in range(num_classes)]
                                
                                # Initialize probability columns
                                for i, class_name in enumerate(class_names):
                                    results_data[f"Prob_{class_name}"] = []
                                
                                # Fill probability values
                                for idx in range(len(csv_df)):
                                    if idx in pred_dict and pred_dict[idx]["Probabilities"]:
                                        probs = pred_dict[idx]["Probabilities"]
                                        for i, class_name in enumerate(class_names):
                                            if i < len(probs):
                                                results_data[f"Prob_{class_name}"].append(probs[i])
                                            else:
                                                results_data[f"Prob_{class_name}"].append(None)
                            
                            # Create results dataframe (only Row, Prediction, and Probabilities)
                            results_df = pd.DataFrame(results_data)
                            
                            # Display results
                            st.success(f"✅ Batch predictions completed for {len(predictions_list)} row(s)!")
                            st.dataframe(results_df, width='stretch')
                            
                            # Download results
                            csv_results = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Batch Results",
                                data=csv_results,
                                file_name="batch_predictions_results.csv",
                                mime="text/csv",
                                help="Download batch prediction results as CSV"
                            )
                            
            except Exception as e:
                st.error(f"**Batch Prediction Error:** {str(e)}\n\nPlease check your CSV file and try again.")

    # Display last prediction if available
    if st.session_state.last_prediction is not None:
        st.markdown("---")
        st.markdown("### Last Prediction Summary")
        pred_label = get_human_readable_label(st.session_state.last_prediction, model_info, label_encoder)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", pred_label)
        if st.session_state.last_proba is not None:
            max_prob = max(st.session_state.last_proba)
            conf_level, _ = get_confidence_level(max_prob)
            with col2:
                st.metric("Confidence", f"{conf_level} ({max_prob:.2%})")

    # Prediction History
    if "prediction_history" in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("## 📜 Prediction History")
        
        history_data = []
        for idx, entry in enumerate(st.session_state.prediction_history[:5]):  # Show last 5
            history_data.append({
                "Index": idx + 1,
                "Inputs": str(entry["inputs"])[:50] + "..." if len(str(entry["inputs"])) > 50 else str(entry["inputs"]),
                "Prediction": entry.get("label", entry.get("prediction", "N/A"))
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, width='stretch')
        
        # Clear history button
        if st.button("🗑️ Clear History", key="clear_history_tab1"):
            st.session_state.prediction_history = []
            st.rerun()


# TAB 2: PROBABILITY ANALYSIS
with tab2:
    st.markdown("## 📊 Probability Analysis")
    
    if st.session_state.last_proba is not None:
        proba = np.array(st.session_state.last_proba)
        
        # Use direct VOC labels mapping (defined at top of file)
        # Determine class names based on number of probabilities
        num_classes = len(proba)
        class_names = [VOC_LABELS.get(i, f"Class_{i}") for i in range(num_classes)]
        
        # Create probability dataframe
        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability": proba
        }).sort_values("Probability", ascending=False)
        
        # Graph 1: Probability Bar Chart using Plotly
        fig = px.bar(
            prob_df, 
            x="Class", 
            y="Probability",
            title="Class Probability Distribution",
            color="Probability",
            color_continuous_scale="Viridis",
            text="Probability"
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(
            yaxis_title="Probability",
            xaxis_title="Class",
            height=500,
            showlegend=False,
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig, width='stretch')
        
        # Display probability table
        st.markdown("### Probability Details")
        prob_df_display = prob_df.copy()
        prob_df_display["Probability"] = prob_df_display["Probability"].apply(lambda x: f"{x:.4f} ({x:.2%})")
        prob_df_display = prob_df_display.reset_index(drop=True)  # Reset index to avoid showing row numbers
        st.dataframe(prob_df_display, width='stretch', hide_index=True)
        
        # Highlight top 3
        st.markdown("### Top 3 Predictions")
        top_3 = prob_df.head(3)
        for idx, row in top_3.iterrows():
            st.markdown(f"- **{row['Class']}**: {row['Probability']:.2%}")
    else:
        st.info("ℹ️ Run a prediction first to see probability analysis. Probability analysis is only available for classification models.")


# TAB 3: FEATURE IMPORTANCE
with tab3:
    st.markdown("## 📈 Feature Importance")
    
    if model_info["has_feature_importances"]:
        importances = model.feature_importances_
        feature_names = [f"Feature {i+1}" for i in range(len(importances))]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        
        # Graph 2: Feature Importance Chart using Plotly
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Feature Importance Ranking",
            color="Importance",
            color_continuous_scale="Blues",
            text="Importance"
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(importance_df) * 40),
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
        
        # Display importance table
        st.markdown("### Feature Importance Details")
        st.dataframe(importance_df, width='stretch')
        
    elif model_info["has_coef"]:
        coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        feature_names = [f"Feature {i+1}" for i in range(len(coef))]
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coef
        }).sort_values("Coefficient", key=abs, ascending=False)
        
        # Graph 2: Coefficient Chart using Plotly
        fig = px.bar(
            importance_df,
            x="Coefficient",
            y="Feature",
            orientation='h',
            title="Feature Coefficients (Linear Model)",
            color="Coefficient",
            color_continuous_scale="RdBu",
            text="Coefficient"
        )
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            xaxis_title="Coefficient",
            yaxis_title="Feature",
            height=max(400, len(importance_df) * 40),
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("### Coefficient Details")
        st.dataframe(importance_df, width='stretch')
    else:
        st.info("ℹ️ This model does not provide feature importance or coefficients.")


# TAB 4: INPUT VISUALIZATION
with tab4:
    st.markdown("## 📉 Input Feature Visualization")
    
    if st.session_state.last_inputs is not None:
        inputs = st.session_state.last_inputs
        feature_names = [f"Feature {i+1}" for i in range(len(inputs))]
        
        # Create input dataframe
        input_df = pd.DataFrame({
            "Feature": feature_names,
            "Raw Value": inputs
        })
        
        # Add scaled values if available
        if st.session_state.last_scaled_inputs is not None:
            input_df["Scaled Value"] = st.session_state.last_scaled_inputs
        
        # Graph 3a: Bar Chart of Input Features
        st.markdown("### Raw Input Values")
        fig = px.bar(
            input_df,
            x="Feature",
            y="Raw Value",
            title="Input Feature Values (Raw)",
            color="Raw Value",
            color_continuous_scale="Plasma",
            text="Raw Value"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')
        
        # Graph 3b: Scaled Values if available
        if st.session_state.last_scaled_inputs is not None:
            st.markdown("### Scaled Input Values")
            fig_scaled = px.bar(
                input_df,
                x="Feature",
                y="Scaled Value",
                title="Input Feature Values (Scaled)",
                color="Scaled Value",
                color_continuous_scale="Viridis",
                text="Scaled Value"
            )
            fig_scaled.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_scaled.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_scaled, width='stretch')
            
            # Comparison chart
            st.markdown("### Raw vs Scaled Comparison")
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=feature_names,
                y=inputs,
                name="Raw Values",
                marker_color='lightblue'
            ))
            fig_comp.add_trace(go.Bar(
                x=feature_names,
                y=st.session_state.last_scaled_inputs,
                name="Scaled Values",
                marker_color='orange'
            ))
            fig_comp.update_layout(
                title="Raw vs Scaled Input Comparison",
                xaxis_title="Feature",
                yaxis_title="Value",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_comp, width='stretch')
        
        # Graph 3c: Radar Chart (Optional)
        if len(inputs) >= 3:
            st.markdown("### Radar Chart Visualization")
            try:
                fig_radar = go.Figure()
                
                # Normalize values for radar chart (0-1 scale)
                normalized_raw = [(v - min(inputs)) / (max(inputs) - min(inputs)) if max(inputs) != min(inputs) else 0.5 for v in inputs]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized_raw + [normalized_raw[0]],  # Close the loop
                    theta=feature_names + [feature_names[0]],  # Close the loop
                    fill='toself',
                    name='Normalized Raw Values',
                    line=dict(color='blue')
                ))
                
                if st.session_state.last_scaled_inputs is not None:
                    scaled = st.session_state.last_scaled_inputs
                    normalized_scaled = [(v - min(scaled)) / (max(scaled) - min(scaled)) if max(scaled) != min(scaled) else 0.5 for v in scaled]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=normalized_scaled + [normalized_scaled[0]],
                        theta=feature_names + [feature_names[0]],
                        fill='toself',
                        name='Normalized Scaled Values',
                        line=dict(color='orange')
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title="Input Feature Radar Chart",
                    height=500
                )
                st.plotly_chart(fig_radar, width='stretch')
            except Exception as e:
                st.warning(f"Could not generate radar chart: {str(e)}")
        
        # Graph 3d: Table Display
        st.markdown("### Input Values Table")
        st.dataframe(input_df, width='stretch')
    else:
        st.info("ℹ️ Run a prediction first to see input visualization.")


# TAB 5: SENSOR ANALYSIS
with tab5:
    st.markdown("## 🔬 Sensor/Feature Analysis")
    
    if st.session_state.last_inputs is not None:
        inputs = np.array(st.session_state.last_inputs)
        feature_names = [f"Feature {i+1}" for i in range(len(inputs))]
        
        # Graph 4: Sensor-specific analysis
        # Detect if features might represent signals (DR, |DR|, EMA, etc.)
        st.markdown("### Feature Signal Analysis")
        
        # Check if we have enough features for time-series-like analysis
        if len(inputs) >= 2:
            # Plot feature pairs if they might represent DR and |DR|
            st.markdown("#### Feature Pair Analysis")
            
            # Create pairs visualization
            if len(inputs) >= 4:
                # Assume first two might be DR-related, next two might be EMA-related
                col1, col2 = st.columns(2)
                
                with col1:
                    # DR vs |DR| plot (if applicable)
                    st.markdown("**Features 1 vs 2**")
                    fig_pair1 = px.scatter(
                        x=[inputs[0]],
                        y=[inputs[1]],
                        title=f"{feature_names[0]} vs {feature_names[1]}",
                        labels={"x": feature_names[0], "y": feature_names[1]}
                    )
                    fig_pair1.update_traces(marker=dict(size=15, color='red'))
                    st.plotly_chart(fig_pair1, width='stretch')
                
                with col2:
                    # EMA-related plot (if applicable)
                    st.markdown("**Features 3 vs 4**")
                    fig_pair2 = px.scatter(
                        x=[inputs[2]],
                        y=[inputs[3]],
                        title=f"{feature_names[2]} vs {feature_names[3]}",
                        labels={"x": feature_names[2], "y": feature_names[3]}
                    )
                    fig_pair2.update_traces(marker=dict(size=15, color='blue'))
                    st.plotly_chart(fig_pair2, width='stretch')
            
            # Time-series-like plot if features could represent a sequence
            st.markdown("#### Feature Sequence Plot")
            fig_ts = px.line(
                x=feature_names,
                y=inputs,
                title="Feature Values as Sequence",
                markers=True
            )
            fig_ts.update_layout(
                xaxis_title="Feature",
                yaxis_title="Value",
                height=400
            )
            st.plotly_chart(fig_ts, width='stretch')
            
            # Statistical summary
            st.markdown("#### Statistical Summary")
            stats_df = pd.DataFrame({
                "Statistic": ["Mean", "Std", "Min", "Max", "Range"],
                "Value": [
                    np.mean(inputs),
                    np.std(inputs),
                    np.min(inputs),
                    np.max(inputs),
                    np.max(inputs) - np.min(inputs)
                ]
            })
            st.dataframe(stats_df, width='stretch')
        else:
            st.info("ℹ️ Need at least 2 features for sensor analysis.")
    else:
        st.info("ℹ️ Run a prediction first to see sensor analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Built with ❤️ using Streamlit</p>
    <p>Production-Ready ML Prediction App v1.0.0</p>
</div>
""", unsafe_allow_html=True)

