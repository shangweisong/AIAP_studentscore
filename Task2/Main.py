import sqlite3
import pandas as pd 
import utils.preprocess as preprocess
import utils.train_models as train_models
import utils.visualisation as vs
from setup import load_config

# load yaml config
config = load_config("config.yaml")

# connect to sql database
db_path = config["database"]["db_path"]
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

# Train models and select the best one
print("Training and selecting the best model...")
best_model, best_model_name, model_results = train_models.train_and_select_model(X_train, y_train, X_test, y_test)

print("Saving the best model...")
train_models.save_model(best_model, "best_model.pkl")

# visualise results
print("Visualizing results...")
vs.plot_r2_scores(model_results)  # Pass model R² results
y_pred = best_model.predict(X_test)
vs.plot_residuals(y_test, y_pred, best_model_name)
vs.plot_predicted_vs_actual(y_test, y_pred, best_model_name)



# if __name__ == "__main__":

#     db_path = '/Users/shangweisong/Desktop/AIAP_Student_score/data/score.db'  # Path to your SQLite database
#     query = "SELECT name FROM sqlite_master WHERE type='table';"  # SQL query to fetch the data

#     print("Starting data preprocessing...")
#     data = preprocess.preprocess_data(db_path, query)  # Ensure correct function call
#     X_train, X_test, y_train, y_test, scaler = preprocess.split_data(data)