# Model Evaluation Report: NYC Taxi Trip Duration

## 1. Executive Summary
This report summarizes the enhancements made to the Taxi Trip Duration prediction pipeline. We implemented Exploratory Data Analysis (EDA), robust outlier detection, and expanded the model suite to include linear regressions (Ridge and Lasso) with feature scaling and cross-validation.

## 2. Exploratory Data Analysis (EDA) Insights

### Target Variable Distribution
The `trip_duration` variable is highly skewed. Applying a logarithmic transformation (`log1p`) results in a near-normal distribution, which is much better suited for most regression algorithms.
- **Mean Duration**: ~955 seconds (~16 minutes).
- **Max Duration**: >2,000,000 seconds (clearly erroneous).

### Spatial Features
Trips were found ranging as far as California (-121 longitude), despite being an NYC dataset. We established strict boundaries for Manhattan/NYC to filter these errors.

## 3. Data Cleaning & Outlier Removal
We implemented a `filter_outliers` function that removes:
- Trips shorter than 1 minute or longer than 12 hours.
- Trips starting or ending outside NYC boundaries:
    - Longitude: [-74.05, -73.75]
    - Latitude: [40.63, 40.85]

## 4. Model Performance Comparison

| Model | Validation RMSE (Log Scale) | Notes |
| :--- | :--- | :--- |
| **XGBoost** | **0.3949** | Best performance, handles non-linearities well. |
| **Ridge** | **0.6189** | Solid baseline, uses L2 regularization and MinMaxScaler. |
| **Lasso** | **0.7950** | Uses L1 regularization, may require alpha tuning. |

*Note: RMSE values are on the log(1 + seconds) scale.*

## 5. Key Technical Implementations

### Feature Scaling
Linear models (Ridge/Lasso) are sensitive to the scale of features. We integrated `MinMaxScaler` into a Scikit-Learn `Pipeline` to ensure all features (Distance, Hour, etc.) are normalized between 0 and 1 before training.

### Cross-Validation
We added `evaluate_with_cv` which uses 5-fold K-Fold Cross Validation. This provides a more robust estimate of how the models will generalize to unseen data.

## 6. Conclusion & Recommendations
- **XGBoost** remains the superior model for this task due to its ability to capture complex spatial and temporal interactions.
- **Outlier removal** significantly improved model stability.
- **Feature Engineering** (Haversine, Manhattan distance) remains the most critical factor in model accuracy.
