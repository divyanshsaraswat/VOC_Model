"""
Helper script to create sample model.pkl, scaler.pkl, and label_encoding.pkl for testing purposes
This script creates all three required .pkl files for testing the Streamlit application
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Generate sample data
print("Generating sample classification dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit scaler
print("Creating and fitting StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit label encoder (optional - converting numeric labels to strings for demonstration)
print("Creating and fitting LabelEncoder...")
label_encoder = LabelEncoder()
# Convert numeric labels to string labels for better demonstration
y_train_labels = [f"Class_{val}" for val in y_train]
y_train_encoded = label_encoder.fit_transform(y_train_labels)

# Train a sample model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_encoded)

# Evaluate
y_test_labels = [f"Class_{val}" for val in y_test]
y_test_encoded = label_encoder.transform(y_test_labels)
score = model.score(X_test_scaled, y_test_encoded)
print(f"Model accuracy: {score:.4f}")

# Save all three files
print("\n💾 Saving files...")
joblib.dump(model, "model.pkl")
print(f"✅ model.pkl saved (Model has {model.n_features_in_} features)")

joblib.dump(scaler, "scaler.pkl")
print(f"✅ scaler.pkl saved (StandardScaler)")

joblib.dump(label_encoder, "label_encoding.pkl")
print(f"✅ label_encoding.pkl saved (LabelEncoder with classes: {list(label_encoder.classes_)})")

print("\n✅ All sample files created successfully!")
print("\n📋 Files created:")
print("   1. model.pkl - Trained RandomForestClassifier")
print("   2. scaler.pkl - StandardScaler for feature preprocessing")
print("   3. label_encoding.pkl - LabelEncoder for prediction labels")
print("\n🚀 You can now run 'streamlit run app.py' to test the application.")
print("\n📝 Note: These are sample files for testing. Replace with your actual trained models for production use.")

