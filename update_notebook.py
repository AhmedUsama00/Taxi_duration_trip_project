import nbformat as nbf
import os

notebook_path = 'notebook.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update imports
import_cell = nb.cells[1]
import_cell.source = """import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# Add src to the path so we can import our modules
sys.path.append(os.path.abspath('src'))

from data.data_loader import load_data, get_target
from features.build_features import preprocess_features
from models.train_model import train_xgboost, train_linear_model, evaluate_with_cv
from models.predict_model import load_model, predict

sns.set_style('whitegrid')
%matplotlib inline"""

# 2. Add EDA and Outlier Detection after Load Data (which is at cell index 3)
eda_markdown = nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis & Outlier Detection")
eda_code = nbf.v4.new_code_cell("""# Check target distribution
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(df_train['trip_duration']), bins=100, kde=True)
plt.title('Distribution of Log(Trip Duration)')
plt.xlabel('Log(Duration + 1)')
plt.show()

# NYC Boundaries
# Longitude: -74.05 to -73.75
# Latitude: 40.63 to 40.85
def filter_outliers(df):
    initial_shape = df.shape[0]
    df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 3600*12)] # 1 min to 12 hours
    df = df[(df['pickup_longitude'] > -74.05) & (df['pickup_longitude'] < -73.75)]
    df = df[(df['pickup_latitude'] > 40.63) & (df['pickup_latitude'] < 40.85)]
    df = df[(df['dropoff_longitude'] > -74.05) & (df['dropoff_longitude'] < -73.75)]
    df = df[(df['dropoff_latitude'] > 40.63) & (df['dropoff_latitude'] < 40.85)]
    print(f"Removed {initial_shape - df.shape[0]} outliers.")
    return df

df_train = filter_outliers(df_train)
df_val = filter_outliers(df_val)""")

# Insert at index 3 (after Load Data code cell)
nb.cells.insert(3, eda_markdown)
nb.cells.insert(4, eda_code)

# 3. Add Linear Models and CV comparison before Prediction
# Prediction was at index 4, but we inserted 2 cells, so it's at index 6 now.
# But wait, "2. Feature Engineering" was at index 7 (now 9).
# "3. Model Training (XGBoost)" was at index 9 (now 11).

linear_md = nbf.v4.new_markdown_cell("## 4. Model Comparison (Ridge, Lasso, CV)")
linear_code = nbf.v4.new_code_cell("""# Train Ridge
model_ridge, rmse_ridge = train_linear_model(X_train, y_train, X_val, y_val, model_type='ridge')

# Train Lasso
model_lasso, rmse_lasso = train_linear_model(X_train, y_train, X_val, y_val, model_type='lasso')

# Cross-Validation comparison (using a sample for speed if necessary)
# cv_results_xgb = evaluate_with_cv(X_train.iloc[:50000], y_train.iloc[:50000], model_type='xgb')
# cv_results_ridge = evaluate_with_cv(X_train.iloc[:50000], y_train.iloc[:50000], model_type='ridge')
""")

# Find where "## 3. Model Training (XGBoost)" is
idx = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and '## 3. Model Training' in cell.source:
        idx = i
        break

# Insert after XGBoost training
nb.cells.insert(idx + 2, linear_md)
nb.cells.insert(idx + 3, linear_code)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook updated successfully.")
