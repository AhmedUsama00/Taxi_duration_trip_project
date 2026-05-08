import pandas as pd
import os

def load_data(data_dir):
    """
    Load the train, validation, and test datasets from the given directory.
    Assumes the presence of train.csv, val.csv, and test.csv inside the 'split' subdirectory.
    """
    train_path = os.path.join(data_dir, 'split', 'train.csv')
    val_path = os.path.join(data_dir, 'split', 'val.csv')
    test_path = os.path.join(data_dir, 'split', 'test.csv')
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_val, df_test

def get_target(df):
    """
    Extracts the target variable 'trip_duration' and drops it from the dataframe.
    """
    if 'trip_duration' in df.columns:
        y = df['trip_duration']
        X = df.drop(columns=['trip_duration'])
        return X, y
    return df, None
