# 🤖 ML Model Predictor - Streamlit Application

This is a production-ready Streamlit application that loads pre-trained .pkl machine learning models and provides an interactive interface for making predictions.

## 📋 Description

This application allows you to interact with trained ML models. You can input features through the UI and get real-time predictions.

### Key Features

- ✅ **Model Loading**: Efficient loading with caching (@st.cache_resource)
- ✅ **Input Validation**: Comprehensive validation for user inputs
- ✅ **Error Handling**: User-friendly error messages
- ✅ **Prediction History**: Track last 10 predictions in session
- ✅ **Probability Scores**: Display confidence scores for classification models
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

4. **Place model file:**
   - Place your trained model file as `model.pkl` in the project directory

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
4. Upload `model.pkl` file

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
│── model.pkl             # Your trained ML model (place here)
│── requirements.txt      # Python dependencies
│── README.md             # Documentation
│── create_sample_model.py  # Helper script to create sample model (optional)
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
- **Preprocessing**: Preprocessing steps should be included in the model or handled separately

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

### Prediction Errors
- Verify input values are within expected range
- Check model expects same number of features as provided
- Ensure input types match model requirements

### Import Errors
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+)

## 📝 Customization Tips

1. **Feature Inputs**: Use specific input types according to your model (sliders, dropdowns, etc.)
2. **Visualizations**: Add feature importance charts
3. **Preprocessing**: Add input preprocessing pipeline
4. **Output Formatting**: Display prediction output in user-friendly format
5. **Dark Mode**: Implement full dark mode support

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

**Note**: This application is generic and works with any scikit-learn compatible model. Customize according to your specific use case.
