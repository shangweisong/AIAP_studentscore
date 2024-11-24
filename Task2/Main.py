import sqlite3
import pandas as pd 
db_path = "/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db"

import preprocess
# import train_models
# import evaluate_model

if __name__ == "__main__":
    db_path = '/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db'  # Path to your SQLite database
    query = "SELECT * FROM your_table_name;"  # SQL query to fetch the data

    # Step 1: Data Preprocessing
    print("Starting data preprocessing...")
    data = preprocess.preprocess_data(db_path, query)
    X_train, X_test, y_train, y_test, scaler = preprocess.split_data(data)