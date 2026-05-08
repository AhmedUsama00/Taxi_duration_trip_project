# NYC Taxi Trip Duration Prediction

## Project Overview
This project predicts the total ride duration of taxi trips in New York City. The dataset includes pickup and dropoff coordinates, timestamps, and passenger counts. The goal of this project is to build an end-to-end Machine Learning pipeline that processes raw data, engineers advanced spatial and temporal features, and trains a highly optimized XGBoost regression model.

## Machine Learning Pipeline Flow

The project is structured into modular Python scripts inside the `src/` directory, following software engineering best practices. 

### 1. Data Ingestion (`src/data/data_loader.py`)
- **`load_data(data_dir)`**: Reads the `train.csv`, `val.csv`, and `test.csv` splits from the dataset directory into Pandas DataFrames.
- **`get_target(df)`**: Separates the target variable (`trip_duration`) from the input features so that the model can learn to predict it.

### 2. Feature Engineering (`src/features/build_features.py`)
Feature engineering is the core of this project, transforming raw timestamps and coordinates into powerful signals for the model.
- **`build_datetime_features(df)`**: Extracts the hour, day of the week, month, and a boolean flag for weekends from the `pickup_datetime`. Traffic patterns heavily depend on the time of day and whether it's a weekday or weekend.
- **`haversine_array(...)`**: Calculates the straight-line "as the crow flies" distance between the pickup and dropoff coordinates.
- **`dummy_manhattan_distance(...)`**: Calculates the Manhattan distance. Since NYC is built on a grid system, taxis must drive along blocks at right angles. This provides a much more accurate proxy for the actual driving distance.
- **`bearing_array(...)`**: Calculates the angle/direction of the trip. Traffic flows differently depending on the direction (e.g., heading North on an avenue vs. East on a cross street).
- **`preprocess_features(df)`**: The master pipeline function that drops unnecessary columns and sequentially applies all the datetime and spatial feature functions above.

### 3. Model Training (`src/models/train_model.py`)
- **`train_xgboost(...)`**: Trains an Extreme Gradient Boosting (XGBoost) model. 
  - **Log Transformation**: Because trip durations are highly skewed (most trips are short, but some take hours), we train the model on the logarithmic scale `log(1 + duration)`. This prevents extreme outliers from throwing off the model.
  - **Evaluation**: The model evaluates its performance using Root Mean Squared Error (RMSE) on the validation set.

### 4. Prediction (`src/models/predict_model.py`)
- **`predict(model, X_test)`**: Feeds the unseen test data into the trained model, and applies the `exp(x) - 1` function to convert the logarithmic predictions back into actual seconds.

---

## How to Improve the Model (Lower RMSE)
*Note: In machine learning regression, a **lower** RMSE means your predictions are closer to the actual values.*

If you want to improve the model further, consider the following techniques:
1. **Hyperparameter Tuning:** Use `GridSearchCV` or `Optuna` to test different XGBoost parameters (like `max_depth`, `learning_rate`, or `n_estimators`).
2. **External Data:** Merge the dataset with NYC weather data (rain/snow slows down traffic) or holiday datasets.
3. **Try LightGBM or CatBoost:** Add another script in `src/models/` for LightGBM. Sometimes ensembling (averaging the predictions of XGBoost and LightGBM) yields the best results.
4. **Geospatial Clustering:** Use KMeans clustering on the pickup coordinates to group neighborhoods together.

---


