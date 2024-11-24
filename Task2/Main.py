import sqlite3
import pandas as pd 
db_path = "/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db"

import preprocess.preprocess_data as preprocess
import load

# import train_models
# import evaluate_model

if __name__ == "__main__":

    db_path = '/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db'  # Path to your SQLite database
    query = "SELECT * FROM your_table_name;"  # SQL query to fetch the data

    data = preprocess.load_data(db_path, query)


#     db_path = '/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db'  # Path to your SQLite database
#     query = "SELECT * FROM your_table_name;"  # SQL query to fetch the data

#     data = preprocess.load_data(db_path, query)
    # X_train, X_test, y_train, y_test, scaler = preprocess.split_data(data)