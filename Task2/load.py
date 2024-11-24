import sqlite3
import pandas as pd

def load_data(db_path,query):
    conn = sqlite3.connect(db_path)
    df= pd.read_sql(query,conn)
    return df
    