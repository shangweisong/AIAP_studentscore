import sqlite3
import pandas as pd 
import utils.preprocess as preprocess

# connect to sql database
db_path = "/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db"
conn = sqlite3.connect(db_path)

# Find table name 
tablename_query = "SELECT name FROM sqlite_master WHERE type='table';"
tablename= pd.read_sql_query(tablename_query, conn)

# Get data from database
df_query = f"SELECT * FROM {tablename.iloc[0,0]};"
df= pd.read_sql_query(df_query, conn)

# Data processing
df_cleaned, scaler= preprocess.preprocess_data(df)
# prepare dataset used for ml

df_ml = preprocess.prepare_dataset(df_cleaned)
X_train, X_test, y_train, y_test,X_train_pca, X_test_pca, y_train_pca, y_test_pca = preprocess.split_data(df_ml)

# print(df_cleaned)
# import train_models
# import evaluate_model

# if __name__ == "__main__":

#     db_path = '/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db'  # Path to your SQLite database
#     query = "SELECT name FROM sqlite_master WHERE type='table';"  # SQL query to fetch the data

#     print("Starting data preprocessing...")
#     data = preprocess.preprocess_data(db_path, query)  # Ensure correct function call
#     X_train, X_test, y_train, y_test, scaler = preprocess.split_data(data)