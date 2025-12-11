# 🧪 Hazardous Gas Classification & Concentration Prediction

This is a production-ready Streamlit application that implements a **two-stage machine learning pipeline** to identify hazardous gases and estimate their concentration (ppmv) from sensor data.

## 📋 System Overview

The application operates in two stages:
1.  **Classification**: Identifies the type of gas (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene) using a classification model.
2.  **Regression**: Estimates the concentration of the identified gas using a specific regression model trained for that gas type.

### Key Features

- ✅ **Two-Stage Prediction**: Seamlessly chains classification and regression models.
- ✅ **Concentration Estimation**: Provides precise ppmv estimates for identified gases.
- ✅ **Rich Visualizations**:
    - **Probability Distribution**: Confidence scores for gas classification.
    - **Feature Importance**: Visualizes top contributing sensors for both classification and concentration.
    - **Sensor Analysis**: Radar charts and heatmaps for raw sensor data.
- ✅ **Batch Processing**: Upload CSV files for bulk prediction (Gas Type + Concentration) with downloadable results.
- ✅ **Prediction History**: Session-based history tracking of classification and concentration results.
- ✅ **Robust Architecture**: Modular design with separate `src` components and organized `models` directory.

## 🚀 Running Locally

### Requirements

- Python 3.8 or higher
- pip package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/voc-streamlit.git
    cd voc-streamlit
    ```

2.  **Create virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Model Setup:**
    Ensure your trained models are in the `models/` directory:
    -   `models/classifier.pkl` (Classifier)
    -   `models/scaler.pkl` (Scaler)
    -   `models/label_encoding.pkl` (Label Encoder)
    -   `models/regressors/*.pkl` (Regression models for each gas)

5.  **Run application:**
    ```bash
    streamlit run app.py
    ```

6.  **Open in browser:**
    -   Default URL: `http://localhost:8501`

## 📁 Project Structure

```
voc-streamlit/
│── app.py                      # Main Streamlit application
│── requirements.txt            # Python dependencies
│── README.md                   # Documentation
│
├── models/                     # Trained ML Models
│   │── classifier.pkl          # Gas classifier
│   │── scaler.pkl              # Input scaler
│   │── label_encoding.pkl      # Label encoder
│   │── regressor_selection_summary.csv # Summary of best regression models
│   └── regressors/             # Individual regression models per gas
│       │── Ethanol_reg.pkl
│       │── Ethylene_reg.pkl
│       └── ...
│
├── notebooks/                  # Jupyter Notebooks for exploration & training
│   │── VOC Classifier.ipynb
│   └── train_regressors.ipynb  # Regression training pipeline
│
├── src/                        # Source Code
│   │── classifier.py           # Classification logic
│   │── data_loader.py          # Data loading & preprocessing
│   │── predict.py              # CLI Prediction script
│   │── regressors.py           # Regression logic
│   │── train_regressors.py     # Regressor training script
│   └── utils.py                # Utility functions (paths, logging)
│
└── assets/                     # Images and static assets
```

## 🛠️ Training & Pipelines

### Training Regressors
To retrain the regression models, use the provided script or notebook:

**Script:**
```bash
python src/train_regressors.py
```
This script will:
1. Load data and preprocess it.
2. Train multiple algorithms (HGBR, Random Forest, Extra Trees, Ridge) for each gas.
3. Select the best model based on RMSE.
4. Save the best models to `models/regressors/`.

**Notebook:**
Open `notebooks/train_regressors.ipynb` for an interactive training session.

### CLI Prediction
You can also run predictions from the command line:

```bash
python src/predict.py --input sample_input.json
```
**Output:**
```json
{
  "gas": "Ethanol",
  "concentration": 10.02
}
```

## 🌐 Deployment

The app is ready for deployment on Streamlit Cloud, HuggingFace Spaces, or any Docker-based environment. Ensure the `models/` directory is included in your deployment build.

## 📄 License
MIT License

## 👤 Author
Created for Advanced Agentic Coding.
