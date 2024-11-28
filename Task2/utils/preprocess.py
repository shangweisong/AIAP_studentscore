import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from sklearn.decomposition import PCA


def preprocess_data(data, config):
    preprocessing_config= config['preprocessing']
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if preprocessing_config['drop_na']:
        data = data.dropna()
    # drop duplicate, drop null and change strings to lowercase
    if preprocessing_config['drop_duplicates']:
        data = data.drop_duplicates()
    
    data = data.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)
    # replace single alphabet replies to yes and no
    data['tuition'] = data['tuition'].replace({'y': 'yes', 'n': 'no'})
    # drop unique identifier (student_id)
    data = data.drop('student_id',axis=1)
    # convert sleep_time and wake_time to sleephours
    data = sleep_hours(data)

    # Standardisation and one-hotencoding
    if preprocessing_config['scale_numerical']:
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    if preprocessing_config['encode_categorical']:
        categorical_cols = data.select_dtypes(include=['object','category']).columns
        data = pd.get_dummies(data, columns= categorical_cols, drop_first=True)
        # convert true and false to 0 and 1
    le = LabelEncoder()
    if 'your_categorical_column' in data.columns:  # Replace with actual column names
        data['your_categorical_column'] = le.fit_transform(data['your_categorical_column'])

    if preprocessing_config['convert_bool']:
        bool_columns = data.select_dtypes(include=['bool']).columns
        data[bool_columns] = data[bool_columns].astype(int)
    # Encode categorical variables if necessary
    
    return data, scaler

def prepare_dataset(data):
    corr_matrix = data.corr()
    final_test_corr = corr_matrix['final_test']
    # Sort the correlations by absolute value (if you want to prioritize strong correlations)
    sorted_corr = final_test_corr.abs().sort_values(ascending=False)
    # Drop column with corr less than 0.1 
    columns_to_drop = sorted_corr[sorted_corr < 0.1].index
    data = data.drop(columns = columns_to_drop, axis =1 )
    # drop datetime columns
    datetime_columns = data.select_dtypes(include=['datetime64']).columns
    data = data.drop(columns=datetime_columns, axis=1)
    return data

def sleep_hours (data):
    # Convert 'sleeptime' and 'waketime' to datetime (using a fixed date to avoid errors)
    data['sleep_time'] = pd.to_datetime(data['sleep_time'], format='%H:%M')
    data['wake_time'] = pd.to_datetime(data['wake_time'], format='%H:%M')

    # Adjust wake_time if it's earlier than sleep_time
    data['wake_time'] = data.apply(
    lambda row: row['wake_time'] + pd.Timedelta(days=1) if row['wake_time'] < row['sleep_time'] else row['wake_time'], axis=1
    )

    # Calculate the difference between wake time and sleep time
    data['sleep_duration'] = (data['wake_time'] - data['sleep_time']).dt.total_seconds() / 3600
    # data['sleep_duration_hours'] = data['sleep_duration'].dt.total_seconds() / 3600
    return data

def pca(data,y ):
    """
    Principal component analysis to reduce features, keep 95% variance
    """
    X = data.drop(columns=['final_test'])  # Replace with actual target column name
    y = data['final_test']  # Replace with actual target column name
    pca = PCA(n_components=0.95)  # Keeps 95% of the variance
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test= train_test_split(X_pca, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def split_data(data):
    X = data.drop(columns=['final_test'])  # Replace with actual target column name
    y = data['final_test']  # Replace with actual target column name
    pca = PCA(n_components=0.95)  # Keeps 95% of the variance
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca= train_test_split(X_pca, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test,X_train_pca, X_test_pca, y_train_pca, y_test_pca

def save_scaler(scaler, filename):
    import joblib
    joblib.dump(scaler, filename)
