"""
Helper script to create a sample model.pkl for testing purposes
यह script एक sample model.pkl file बनाने के लिए है testing के लिए
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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

# Train a sample model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.4f}")

# Save model
model_filename = "model.pkl"
joblib.dump(model, model_filename)
print(f"✅ Model saved as '{model_filename}'")
print(f"Model has {model.n_features_in_} features")
print(f"Model classes: {model.classes_}")

# Save sample model for regression too (optional)
from sklearn.ensemble import RandomForestRegressor

print("\nGenerating sample regression dataset...")
X_reg, y_reg = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=1,
    random_state=42
)
y_reg = y_reg.ravel()

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("Training Random Forest Regressor...")
model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model_reg.fit(X_train_reg, y_train_reg)

score_reg = model_reg.score(X_test_reg, y_test_reg)
print(f"Model R² score: {score_reg:.4f}")

# Uncomment below to save regression model instead
# joblib.dump(model_reg, "model_regression.pkl")
# print("✅ Regression model saved as 'model_regression.pkl'")

print("\n✅ Sample model created successfully!")
print("You can now run 'streamlit run app.py' to test the application.")

