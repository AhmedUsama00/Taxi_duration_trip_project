import numpy as np
import pandas as pd

def haversine_array(lat1, lng1, lat2, lng2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    """
    Calculate Manhattan distance (sum of N/S and E/W distances)
    """
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    """
    Calculate the bearing/direction between two points.
    """
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def build_datetime_features(df):
    """
    Extract datetime features from pickup_datetime.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['pickup_datetime']):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['is_weekend'] = (df['pickup_datetime'].dt.dayofweek >= 5).astype(int)
    
    # Drop original datetime string if desired
    # df = df.drop(columns=['pickup_datetime'])
    return df

def build_spatial_features(df):
    """
    Compute Haversine, Manhattan, and Bearing features.
    """
    df['distance_haversine'] = haversine_array(
        df['pickup_latitude'].values, df['pickup_longitude'].values,
        df['dropoff_latitude'].values, df['dropoff_longitude'].values
    )
    
    df['distance_dummy_manhattan'] = dummy_manhattan_distance(
        df['pickup_latitude'].values, df['pickup_longitude'].values,
        df['dropoff_latitude'].values, df['dropoff_longitude'].values
    )
    
    df['direction'] = bearing_array(
        df['pickup_latitude'].values, df['pickup_longitude'].values,
        df['dropoff_latitude'].values, df['dropoff_longitude'].values
    )
    return df

def preprocess_features(df):
    """
    Full feature engineering pipeline.
    """
    # Drop columns that shouldn't be used for modeling
    cols_to_drop = ['id', 'store_and_fwd_flag', 'dropoff_datetime']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    df = build_datetime_features(df)
    df = build_spatial_features(df)
    
    # We drop the datetime object to keep the dataframe purely numeric for many ML models
    if 'pickup_datetime' in df.columns:
        df = df.drop(columns=['pickup_datetime'])
        
    return df
