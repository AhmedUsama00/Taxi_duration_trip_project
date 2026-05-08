import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('src'))
from data.data_loader import load_data, get_target
from features.build_features import preprocess_features
from models.train_model import train_linear_model, evaluate_with_cv

data_dir = 'data_set'
df_train, df_val, df_test = load_data(data_dir)

# Sample for quick test
df_train = df_train.iloc[:10000]
df_val = df_val.iloc[:2000]

X_train, y_train = get_target(df_train)
X_val, y_val = get_target(df_val)

X_train = preprocess_features(X_train)
X_val = preprocess_features(X_val)

print("Testing Ridge...")
model_ridge, rmse_ridge = train_linear_model(X_train, y_train, X_val, y_val, model_type='ridge')

print("Testing Lasso...")
model_lasso, rmse_lasso = train_linear_model(X_train, y_train, X_val, y_val, model_type='lasso')

print("Testing CV...")
cv_scores = evaluate_with_cv(X_train, y_train, model_type='ridge', n_splits=3)
