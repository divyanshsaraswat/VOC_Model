# 🎨 Customization Guide

## Customize Application According to Your Model

### 📝 Main Customization Points

## 1. Customizing Feature Input Fields

Modify the `create_default_inputs()` function in `app.py`:

### Example 1: Specific Feature Names and Types

```python
def create_default_inputs(n_features=4):
    inputs = []
    
    # Feature 1: Age (Slider)
    age = st.slider(
        "Age",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        key="age"
    )
    inputs.append(age)
    
    # Feature 2: Income (Number Input)
    income = st.number_input(
        "Annual Income ($)",
        min_value=0.0,
        max_value=1000000.0,
        value=50000.0,
        step=1000.0,
        key="income"
    )
    inputs.append(income)
    
    # Feature 3: Category (Dropdown)
    category = st.selectbox(
        "Category",
        options=["Type A", "Type B", "Type C"],
        key="category"
    )
    # Encode category (0 for Type A, 1 for Type B, 2 for Type C)
    category_encoded = ["Type A", "Type B", "Type C"].index(category)
    inputs.append(category_encoded)
    
    # Feature 4: Text Input (if needed)
    text_input = st.text_input(
        "Description",
        value="",
        key="description"
    )
    # Process text input (e.g., length, word count)
    text_length = len(text_input.split())
    inputs.append(text_length)
    
    return inputs
```

### Example 2: Dynamic Feature Names

```python
def create_default_inputs(n_features=4):
    inputs = []
    
    # Define feature names and ranges
    feature_configs = [
        {"name": "Temperature", "min": -50, "max": 50, "default": 25},
        {"name": "Humidity", "min": 0, "max": 100, "default": 60},
        {"name": "Pressure", "min": 900, "max": 1100, "default": 1013},
        {"name": "Wind Speed", "min": 0, "max": 100, "default": 10},
    ]
    
    for i, config in enumerate(feature_configs[:n_features]):
        value = st.number_input(
            config["name"],
            min_value=float(config["min"]),
            max_value=float(config["max"]),
            value=float(config["default"]),
            step=0.1,
            key=f"feature_{i}"
        )
        inputs.append(value)
    
    return inputs
```

## 2. Adding Preprocessing Steps

If your model requires preprocessing:

```python
def preprocess_inputs(inputs):
    """
    Preprocess inputs before prediction
    """
    # Example: Scaling
    from sklearn.preprocessing import StandardScaler
    
    # Load your scaler (if saved)
    # scaler = joblib.load('scaler.pkl')
    # inputs_scaled = scaler.transform([inputs])
    
    # Example: Feature engineering
    # feature1_squared = inputs[0] ** 2
    # feature_interaction = inputs[0] * inputs[1]
    # processed_inputs = inputs + [feature1_squared, feature_interaction]
    
    # For now, return as-is
    return inputs

# In prediction section:
processed_inputs = preprocess_inputs(feature_inputs)
input_array = np.array(processed_inputs).reshape(1, -1)
prediction = model.predict(input_array)[0]
```

## 3. Customizing Output Formatting

### Display with Friendly Labels

```python
# After prediction
if isinstance(prediction, (np.integer, int)):
    prediction_label = get_prediction_label(prediction)
    st.success(f"✅ Prediction: **{prediction_label}**")
else:
    st.success(f"✅ Prediction: **{prediction:.4f}**")

def get_prediction_label(prediction):
    """Convert numeric prediction to friendly label"""
    labels = {
        0: "Not Spam",
        1: "Spam",
        2: "Unknown"
    }
    return labels.get(prediction, str(prediction))
```

### Display Classification Results

```python
# If classification model
if model_info["classes"]:
    predicted_class = model_info["classes"][prediction]
    
    st.markdown(f"### Predicted Class: **{predicted_class}**")
    
    # Show top 3 probabilities
    top_3_indices = np.argsort(proba)[-3:][::-1]
    st.markdown("### Top 3 Predictions:")
    
    for idx in top_3_indices:
        class_name = model_info["classes"][idx]
        probability = proba[idx]
        st.markdown(f"- **{class_name}**: {probability:.2%}")
```

## 4. Customizing Sidebar Content

Add specific information about your model:

```python
with st.sidebar:
    st.markdown("## 📋 About This Model")
    
    st.markdown("""
    ### Model Information
    This model predicts house prices based on:
    - Square footage
    - Number of bedrooms
    - Location
    - Age of property
    
    ### Training Data
    - Dataset: Kaggle House Prices
    - Samples: 10,000+
    - Features: 20
    
    ### Performance
    - Accuracy: 95.2%
    - RMSE: $15,000
    
    ### Limitations
    - Only works for properties < $1M
    - Location limited to US cities
    """)
```

## 5. Adding Visualizations

### Feature Importance Chart

```python
# If model supports feature importances
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = [f"Feature {i+1}" for i in range(len(importances))]
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    st.bar_chart(importance_df.set_index("Feature"))
```

### Input Distribution Visualization

```python
# Display input distribution
input_df = pd.DataFrame({
    "Feature": [f"Feature {i+1}" for i in range(len(feature_inputs))],
    "Value": feature_inputs
})

st.bar_chart(input_df.set_index("Feature")["Value"])
```

## 6. Multiple Models Support

If you want to support multiple models:

```python
# In sidebar
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=["Model A (Classification)", "Model B (Regression)"],
    key="model_selector"
)

# Load appropriate model
if selected_model == "Model A (Classification)":
    model, error = load_model("model_classification.pkl")
else:
    model, error = load_model("model_regression.pkl")
```

## 7. File Upload Support

For batch predictions:

```python
uploaded_file = st.file_uploader(
    "Upload CSV file for batch predictions",
    type=["csv"],
    help="CSV file should have columns matching input features"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if st.button("Predict All"):
        predictions = model.predict(df.values)
        df['Prediction'] = predictions
        st.dataframe(df)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )
```

## 8. Advanced Error Handling

```python
def validate_and_prepare_inputs(inputs, model_info):
    """
    Advanced validation and preparation
    """
    errors = []
    warnings = []
    
    # Check feature count
    if len(inputs) != model_info["n_features"]:
        errors.append(f"Expected {model_info['n_features']} features")
    
    # Check value ranges (if known)
    valid_ranges = {
        0: (-100, 100),  # Feature 0 range
        1: (0, 1000),    # Feature 1 range
        # ... define for all features
    }
    
    for i, value in enumerate(inputs):
        if i in valid_ranges:
            min_val, max_val = valid_ranges[i]
            if value < min_val or value > max_val:
                warnings.append(
                    f"Feature {i} ({value}) is outside typical range [{min_val}, {max_val}]"
                )
    
    # Check for outliers
    mean_values = [0.5, 0.5, 0.5, 0.5]  # Your typical means
    std_values = [0.1, 0.1, 0.1, 0.1]   # Your typical stds
    
    for i, value in enumerate(inputs):
        if i < len(mean_values) and i < len(std_values):
            z_score = abs((value - mean_values[i]) / std_values[i])
            if z_score > 3:
                warnings.append(f"Feature {i} may be an outlier (z-score: {z_score:.2f})")
    
    return errors, warnings
```

## 9. Model Explanation

Add SHAP values:

```python
# Add to requirements.txt: shap

import shap

# Load explainer
@st.cache_resource
def load_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer

# In prediction section:
if st.checkbox("Show Feature Importance"):
    explainer = load_explainer(model)
    shap_values = explainer.shap_values(input_array)
    st.write("Feature Importance (SHAP):")
    # Display SHAP values
```

## 10. Theme Customization

Create `.streamlit/config.toml` file:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

---

## 📋 Quick Reference Checklist

- [ ] Feature names and types customized
- [ ] Input validation rules added
- [ ] Preprocessing steps added
- [ ] Output formatting customized
- [ ] Sidebar information updated
- [ ] Visualizations added
- [ ] Error messages customized
- [ ] Model-specific help text added

---

**💡 Tips:**

1. Check your model's training code - see preprocessing steps there
2. Document feature names and expected ranges
3. Test thoroughly before deployment
4. Collect user feedback and improve

**🎯 Remember:** It's essential to customize the generic implementation according to your specific use case!
