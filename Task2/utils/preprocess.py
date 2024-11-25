import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np



def preprocess_data(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")


    # drop duplicate, drop null and change strings to lowercase
    data = data.dropna()
    data = data.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)
    data = data.drop_duplicates()
    # replace single alphabet replies to yes and no
    data['tuition'] = data['tuition'].replace({'y': 'yes', 'n': 'no'})


    # Standardisation and one-hotencoding
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Identify categorical columns and one-hot encoding subsequently
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    data = pd.get_dummies(data, columns= categorical_cols, drop_first=True)
    # convert true and false to 0 and 1
    bool_columns = data.select_dtypes(include=['bool']).columns
    data[bool_columns] = data[bool_columns].astype(int)
    
    # # Encode categorical variables if necessary
    # le = LabelEncoder()
    # if 'your_categorical_column' in data.columns:  # Replace with actual column names
    #     data['your_categorical_column'] = le.fit_transform(data['your_categorical_column'])
    
    return data, scaler
def prepare_dataset(data):
    # corr_matrix = data.corr()
    # final_test_corr = corr_matrix['final_test']
    # # Sort the correlations by absolute value (if you want to prioritize strong correlations)
    # sorted_corr = final_test_corr.abs().sort_values(ascending=False)
    # # Drop column with corr less than 0.1 
    # columns_to_drop = sorted_corr[sorted_corr < 0.1].index
    # data = data.drop(columns = columns_to_drop, axis =1 )
    print("Step 1: Checking for 'final_test' column...")
    if 'final_test' not in data.columns:
        raise ValueError("'final_test' column is missing from the dataset.")
    
    print("Step 2: Calculating correlation matrix...")
    corr_matrix = data.corr()

    print("Step 3: Extracting correlations for 'final_test'...")
    final_test_corr = corr_matrix['final_test']

    print("Step 4: Sorting correlations by absolute value...")
    sorted_corr = final_test_corr.abs().sort_values(ascending=False)

    print("Step 5: Identifying columns to drop...")
    columns_to_drop = sorted_corr[sorted_corr < 0.1].index

    print(f"Columns to drop: {list(columns_to_drop)}")
    print("Step 6: Dropping columns...")
    data = data.drop(columns=columns_to_drop, axis=1)

    print("Step 7: Returning dataset...")
    return data

def split_data(data):
    X = data.drop(columns=['final_test'])  # Replace with actual target column name
    y = data['final_test']  # Replace with actual target column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def save_scaler(scaler, filename):
    import joblib
    joblib.dump(scaler, filename)
