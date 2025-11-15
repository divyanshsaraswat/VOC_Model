# 🤖 ML Model Predictor - Streamlit Application

This is a production-ready Streamlit application that loads pre-trained .pkl machine learning models and scalers, providing an interactive interface for making predictions with proper preprocessing.

## 📋 Description

This application allows you to interact with trained ML models. You can input features through the UI and get real-time predictions.

### Key Features

- ✅ **Model & Scaler Loading**: Efficient loading of both `model.pkl` and `scaler.pkl` with caching (@st.cache_resource)
- ✅ **Automatic Preprocessing**: Scales inputs using scaler before prediction (matches training pipeline)
- ✅ **Input Validation**: Comprehensive validation for user inputs
- ✅ **Error Handling**: User-friendly error messages with graceful fallback if scaler is missing
- ✅ **Rich Visualizations**: Interactive charts using Plotly (probability distributions, feature importance, input analysis)
- ✅ **Prediction History**: Track last 10 predictions in session
- ✅ **Probability Scores**: Display confidence scores for classification models with High/Medium/Low indicators
- ✅ **Multi-Tab Interface**: Organized tabs for Prediction, Probability Analysis, Feature Importance, Input Visualization, and Sensor Analysis
- ✅ **Responsive UI**: Clean, professional, and user-friendly interface
- ✅ **Production Ready**: Optimized for deployment on Streamlit Cloud, HuggingFace Spaces, Render, Railway

## 🚀 Running Locally

### Requirements

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/voc-streamlit.git
cd voc-streamlit
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Place model files:**
   - Place your trained model file as `model.pkl` in the project directory (required)
   - Place your scaler file as `scaler.pkl` in the project directory (optional but recommended)
   - Place your label encoder file as `label_encoding.pkl` in the project directory (optional but recommended)
   - **Note**: 
     - `model.pkl` is required - app will not start without it
     - If `scaler.pkl` is not available, the app will run predictions without scaling
     - If `label_encoding.pkl` is not available, the app will use numeric predictions or model classes

5. **Run application:**
```bash
streamlit run app.py
```

6. **Open in browser:**
   - Application will open automatically in your browser
   - Default URL: `http://localhost:8501`

## 🌐 Deployment

### Streamlit Cloud

1. Push repository to GitHub
2. Login to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select repository and specify `app.py`
5. Deploy!

**Streamlit Cloud automatically detects `requirements.txt`**

### HuggingFace Spaces

1. Create new Space on [HuggingFace Spaces](https://huggingface.co/spaces)
2. Select Streamlit SDK
3. Ensure `requirements.txt` is in repository
4. Upload `model.pkl` file (required)
5. Upload `scaler.pkl` file (optional but recommended for proper preprocessing)

### Render / Railway

**Render:**

1. Create new Web Service in Render dashboard
2. Connect repository
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Environment: Python 3

**Railway:**

1. Create new service in Railway project
2. Connect GitHub repository
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy!

## 📁 File Structure

```
voc-streamlit/
│── app.py                 # Main Streamlit application
│── model.pkl             # Your trained ML model (required)
│── scaler.pkl            # Your scaler/preprocessor (optional but recommended)
│── label_encoding.pkl    # Your label encoder (optional but recommended)
│── requirements.txt      # Python dependencies
│── README.md             # Documentation
│── create_sample_model.py  # Helper script to create sample files (optional)
└── assets/               # Optional: images, icons, etc.
```

## 🎯 Example Usage

1. **Open application** - Run Streamlit app
2. **Input features** - Enter values for all required features
3. **Click Predict** - Get prediction
4. **View results** - See prediction and confidence scores
5. **Check history** - View previous predictions

### Example Input

If your model has 4 features:
- Feature 1: 10.5
- Feature 2: 20.3
- Feature 3: 15.7
- Feature 4: 8.9

**Output:**
- Predicted Value: [model prediction]
- Confidence Scores: [probability distribution if classification]

## 🛠️ Model Training Notes

### Model Requirements

Your model file should be in the following format:

- **Format**: `.pkl` file created with `joblib.dump()` or `pickle.dump()`
- **Compatibility**: scikit-learn models, custom models with `.predict()` method
- **Preprocessing**: The app supports separate scaler file for proper preprocessing

### Scaler Requirements

The application supports using a separate scaler file for preprocessing:

- **File**: `scaler.pkl` (must have `.transform()` method)
- **Compatibility**: StandardScaler, MinMaxScaler, RobustScaler, or any sklearn scaler
- **Usage**: Inputs are automatically scaled using `scaler.transform()` before prediction
- **Optional**: App works without scaler, but scaling ensures predictions match training pipeline
- **Loading**: Uses `joblib.load()` with caching for performance

### Label Encoder Requirements

The application supports using a separate label encoder file for converting numeric predictions to human-readable labels:

- **File**: `label_encoding.pkl`
- **Compatibility**: 
  - sklearn `LabelEncoder` (with `inverse_transform()` method)
  - Dictionary mapping `{numeric_value: label_name}`
- **Usage**: Predictions are automatically converted to labels using `label_encoder.inverse_transform()` or dict lookup
- **Optional**: App works without label encoder, but labels improve user experience
- **Loading**: Uses `joblib.load()` with caching for performance
- **Priority**: Label encoder > Model classes > Raw prediction value

**How it works:**
1. User inputs features
2. App converts inputs to numpy array
3. **If scaler exists**: `scaled_inputs = scaler.transform(input_array)`
4. **If no scaler**: Uses raw inputs
5. Prediction: `prediction = model.predict(scaled_inputs)` (numeric value)
6. **If label encoder exists**: `label = label_encoder.inverse_transform(prediction)`
7. **If no label encoder**: Uses model classes or raw prediction value

### Customizing Input Fields

In `app.py`, modify the `create_default_inputs()` function to customize input fields according to your model:

```python
def create_default_inputs(n_features=4):
    # Customize input types here
    # Example: Use sliders, dropdowns, text inputs based on feature type
    inputs = []
    # Your custom input logic
    return inputs
```

### Feature Names

Customize input labels according to your model's feature names:

```python
st.number_input(
    "Your Feature Name",  # Change this
    # ... other parameters
)
```

## ⚠️ Limitations

1. **Model Size**: Very large models (>100MB) may take time to load
2. **Input Validation**: All features should be numeric (customization required for other types)
3. **Preprocessing**: Model training time preprocessing steps may need to be manually applied
4. **Session State**: Prediction history is session-specific (cleared on page refresh)

## 🐛 Troubleshooting

### Model File Not Found
- Ensure `model.pkl` file exists in project root
- Check file name spelling
- App will not start without `model.pkl`

### Scaler File Not Found
- App will show an info message if `scaler.pkl` is missing
- Predictions will still work using unscaled inputs
- For best results, ensure `scaler.pkl` is present if your model was trained with scaled features

### Label Encoder File Not Found
- App will show an info message if `label_encoding.pkl` is missing
- Predictions will still work using numeric values or model classes
- For better user experience, ensure `label_encoding.pkl` is present for readable labels
- The app falls back to `model.classes_` or raw prediction values if encoder is missing

### Scaling Errors
- If scaling fails, app will fall back to unscaled inputs
- Verify scaler file is compatible (should have `.transform()` method)
- Check that scaler was trained on same number of features as your model
- Ensure scaler file is not corrupted

### Prediction Errors
- Verify input values are within expected range
- Check model expects same number of features as provided
- Ensure input types match model requirements
- If using scaler, verify scaling is successful (check error messages)

### Import Errors
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+)
- Ensure `plotly` is installed for visualizations

## 📝 Customization Tips

1. **Feature Inputs**: Use specific input types according to your model (sliders, dropdowns, etc.)
2. **Scaler Path**: If your scaler has a different name/path, modify `load_scaler()` function in `app.py`
3. **Visualizations**: Rich Plotly visualizations included (probability charts, feature importance, input analysis, sensor graphs)
4. **Preprocessing**: Scaler is automatically applied - ensure it matches your training pipeline
5. **Output Formatting**: Prediction cards with confidence levels and pass/fail indicators (customize thresholds in code)
6. **Tabs**: Five organized tabs for different analysis views (customize based on your needs)

### Saving Your Model, Scaler, and Label Encoder

When training your model, save all three files:

```python
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Prepare data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Encode labels if needed (for classification)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train model
model.fit(X_train_scaled, y_train_encoded)

# Save all three files
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoding.pkl')

print("✅ All files saved successfully!")
print(f"   - model.pkl: Trained model")
print(f"   - scaler.pkl: Feature scaler")
print(f"   - label_encoding.pkl: Label encoder")
```

**Alternative: Using Dictionary for Label Encoding**

If you prefer a dictionary mapping instead of LabelEncoder:

```python
# Create label mapping
label_mapping = {0: "Class A", 1: "Class B", 2: "Class C"}
joblib.dump(label_mapping, 'label_encoding.pkl')
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## 📄 License

MIT License - Feel free to use this project

## 👤 Author

Created with ❤️ for ML practitioners

---

**Note**: This application is generic and works with any scikit-learn compatible model. It supports three .pkl files:
- **model.pkl** (required): Your trained ML model
- **scaler.pkl** (optional): Feature preprocessor/scaler  
- **label_encoding.pkl** (optional): Label encoder for human-readable predictions

For best results, use all three files that match your training pipeline. Customize according to your specific use case.
