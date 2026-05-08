import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

def train_xgboost(X_train, y_train, X_val, y_val, model_dir='models'):
    """
    Train an XGBoost model on the dataset.
    Trip durations are highly skewed, so we typically train on log(1 + duration).
    """
    # Transform target to log scale
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    print("Training XGBoost Model...")
    
    # Define hyperparams
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.1,
        'max_depth': 8,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    
    # Fit the model
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
        verbose=10
    )
    
    # Evaluate
    val_preds_log = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val_log, val_preds_log))
    print(f"Validation RMSE (log scale): {rmse:.4f}")
    
    # Save the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, 'xgb_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, rmse

def train_linear_model(X_train, y_train, X_val, y_val, model_type='ridge', alpha=1.0, model_dir='models'):
    """
    Train a linear model (Ridge or Lasso) with MinMaxScaler.
    """
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    print(f"Training {model_type.capitalize()} Model...")
    
    if model_type == 'ridge':
        model_obj = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model_obj = Lasso(alpha=alpha)
    else:
        raise ValueError("model_type must be 'ridge' or 'lasso'")
        
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('regressor', model_obj)
    ])
    
    pipeline.fit(X_train, y_train_log)
    
    # Evaluate
    val_preds_log = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val_log, val_preds_log))
    print(f"Validation RMSE ({model_type}, log scale): {rmse:.4f}")
    
    # Save the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    return pipeline, rmse

def evaluate_with_cv(X, y, model_type='xgb', n_splits=5):
    """
    Perform K-Fold Cross Validation.
    """
    y_log = np.log1p(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if model_type == 'xgb':
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 8,
            'n_estimators': 100, # Fewer for speed in CV
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)
    elif model_type == 'ridge':
        model = Pipeline([('scaler', MinMaxScaler()), ('regressor', Ridge())])
    elif model_type == 'lasso':
        model = Pipeline([('scaler', MinMaxScaler()), ('regressor', Lasso())])
    
    print(f"Performing {n_splits}-fold CV for {model_type}...")
    scores = cross_val_score(model, X, y_log, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)
    rmse_scores = -scores
    print(f"CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
    return rmse_scores
