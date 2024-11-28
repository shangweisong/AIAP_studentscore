from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import joblib

def train_and_select_model(X_train, y_train, X_test, y_test,config):
    
    models_config = config["models"]
    best_model = None
    best_model_name = None
    best_r2_score = float('-inf')  # Start with a very low score
    lowest_mse= float('inf')
    model_results=[]

    for model_name, model_info in models_config.items():
        if not model_info["enabled"]:
            print(f"Skipping {model_name} (disabled in config).")
            continue
     
        print(f"Training {model_name}...")
        model = eval(f"{model_name.replace(' ', '')}()")  # Dynamically initialize model
        param_grid = model_info["params"]

        if param_grid:  # Perform GridSearch only if there are parameters to tune
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='r2',
                cv=3,  # 3-fold cross-validation
                n_jobs=-1  # Use all available cores
            )
            grid_search.fit(X_train, y_train)
            tuned_model = grid_search.best_estimator_
            print(f"Best params for {model_name}: {grid_search.best_params_}")
        else:
            # If no hyperparameters to tune, train the base model
            tuned_model = model
            tuned_model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = tuned_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse= mean_squared_error(y_test,y_pred)
        print(f"{model_name} R^2: {r2:.4f} MSE: {mse:.4f}")
        model_results.append((model_name, r2, mse))


        if r2 > best_r2_score or (r2 == best_r2_score and mse < lowest_mse):
            best_r2_score = r2
            lowest_mse = mse
            best_model = tuned_model
            best_model_name = model_name    
            
    # Return the best model based on R^2 score
    print(f"Best Model: {best_model_name} with R^2: {best_r2_score:.4f} and MSE: {lowest_mse:.4f}")
    return best_model, best_model_name, model_results

def save_model(model, filename):
    joblib.dump(model, filename)