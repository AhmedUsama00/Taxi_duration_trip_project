import numpy as np
import joblib
import pandas as pd
import os

def load_model(model_path='models/xgb_model.pkl'):
    """
    Load the trained XGBoost model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    return model

def predict(model, X_test):
    """
    Generate predictions using the trained model.
    Converts predictions back from log scale.
    """
    print("Generating predictions...")
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    return preds
