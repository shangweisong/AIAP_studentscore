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


# print (df)
# Data processing
df_cleaned, scaler= preprocess.preprocess_data(df)

print("Finding correlation")
corr= df_cleaned.corr()
print("Step 3: Extracting correlations for 'final_test'...")
final_test_corr = corr['final_test']
print("Step 4: Sorting correlations by absolute value...")
sorted_corr = final_test_corr.abs().sort_values(ascending=False)
print("Step 5: Identifying columns to drop...")
columns_to_drop = sorted_corr[sorted_corr < 0.1].index
print(f"Columns to drop: {list(columns_to_drop)}")
print("Step 6: Dropping columns...")
df_ml = df_cleaned.drop(columns=columns_to_drop, axis=1)

print(df_ml)
# df_ml = preprocess.prepare_dataset(df_cleaned)
# print(df_ml)

# import train_models
# import evaluate_model

# if __name__ == "__main__":

#     db_path = '/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db'  # Path to your SQLite database
#     query = "SELECT name FROM sqlite_master WHERE type='table';"  # SQL query to fetch the data

#     print("Starting data preprocessing...")
#     data = preprocess.preprocess_data(db_path, query)  # Ensure correct function call
#     X_train, X_test, y_train, y_test, scaler = preprocess.split_data(data)