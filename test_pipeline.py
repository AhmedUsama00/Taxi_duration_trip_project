import sys
import os
sys.path.append(os.path.abspath('src'))

from data.data_loader import load_data, get_target
from features.build_features import preprocess_features
from models.train_model import train_xgboost
from models.predict_model import load_model, predict
import pandas as pd

print("Loading data...")
df_train = pd.read_csv('data_set/split/train.csv', nrows=1000)
df_val = pd.read_csv('data_set/split/val.csv', nrows=100)
df_test = pd.read_csv('data_set/split/test.csv', nrows=100)

print("Preprocessing...")
X_train, y_train = get_target(df_train)
X_val, y_val = get_target(df_val)
X_test, _ = get_target(df_test)

X_train = preprocess_features(X_train)
X_val = preprocess_features(X_val)
X_test = preprocess_features(X_test)

print("Training...")
model, rmse = train_xgboost(X_train, y_train, X_val, y_val, model_dir='models_test')

print("Predicting...")
preds = predict(model, X_test)
print(f"Predictions: {preds[:5]}")
print("Pipeline Test Passed!")
