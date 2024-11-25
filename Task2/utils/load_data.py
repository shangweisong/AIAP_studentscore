import sqlite3
import pandas as pd

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    tablename_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tablename= pd.read_sql_query(tablename_query, conn)
    df_query = f"SELECT * FROM {tablename.iloc[0,0]};"
    df= pd.read_sql_query(df_query, conn)
    return df