# Run this cell to import the modules you require
import mlflow.sklearn
import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import mlflow


def entry_point():
    """
    Main entry point for the training script.
    This function orchestrates the loading of data, preprocessing, model training, and evaluation.
    """
    # Read in the data

    original_data = load_data("./data/london_weather.csv")
    cleaned_data = clean_up_data(original_data)
    important_features = ['min_temp', 'global_radiation', 'month', 'cloud_cover']
    X_train, X_test, y_train, y_test = split_data(cleaned_data, important_features)
    

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("london_weather_experiment", )
    # mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.log_param("features", X_train.columns.tolist())
        mlflow.log_param("target", "mean_temp")
        trained_model = train_linear_regression_model(X_train, y_train)
        evaluate_model(trained_model, X_test, y_test)
    

def load_data(file_path):
    """
    Load the weather data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file containing weather data.
    
    Returns:
    DataFrame: The loaded weather data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def clean_up_data(weather):
    """
    Clean up the data by handling missing values and scaling features.
    
    Parameters:
    data (DataFrame): The input data to clean.
    
    Returns:
    DataFrame: The cleaned data.
    """
    # Copy data.
    cleaned_weather = weather.copy()
    
    # Identify columns with missing value rate less than 0.05%
    cleaned_weather['date'] = pd.to_datetime(cleaned_weather['date'], format='%Y%m%d')
    missing_rates = cleaned_weather.isna().sum() / len(cleaned_weather) * 100
    cols_to_dropna = missing_rates[missing_rates < 0.05].index.tolist()

    # Remove rows with missing values in those columns
    cleaned_weather = cleaned_weather.dropna(subset=cols_to_dropna)
    
    # Impute missing values using interpolation
    missing_cols = cleaned_weather.columns[cleaned_weather.isna().any()].tolist()
    for col in missing_cols:
        if cleaned_weather[col].isna().sum() > 0:
            cleaned_weather[col].interpolate(method='linear', inplace=True)
    
    cleaned_weather.set_index('date', inplace=True)
    cleaned_weather['month'] = cleaned_weather.index.month
    cleaned_weather['year'] = cleaned_weather.index.year
    
    return cleaned_weather

def split_data(cleaned_weather, important_features):
    """
    Split the data into training and testing sets.
    
    Parameters:
    cleaned_weather (DataFrame): The cleaned weather data.
    important_features (list): List of features to be used for training.
    
    Returns:
    tuple: Training and testing sets for features and target variable.
    """
    X = cleaned_weather[important_features]
    y = cleaned_weather['mean_temp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test

def build_preprocess_steps():
    """
    Preprocess the data by scaling numerical features.
    
    Parameters:
    data (DataFrame): The input data to preprocess.
    
    Returns:
    DataFrame: The preprocessed data.
    """
    steps = [('scaler', StandardScaler())]
    return steps

def train_linear_regression_model(X_train, y_train):
    """
    Train a logistic regression model.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target variable.
    
    Returns:
    LinearRegression: The trained logistic regression model.
    """

    steps = build_preprocess_steps()
    steps.append(('linear_regression', LinearRegression()))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    cv_result = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    mlflow.log_metric("cross_val_mse", -cv_result.mean())
    mlflow.log_metric("cross_val_mse_std", cv_result.std())

    print(f"Cross-validated MSE: {-cv_result.mean()} (+/- {cv_result.std()})")
    # print(f"Best Estimator: {pipeline.named_steps['linear_regression']}")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_train)
    trained_mse = mean_squared_error(y_train, predictions)
    print("Validated MSE on training set:", trained_mse)
    mlflow.log_metric("trained_mse", trained_mse)
    mlflow.sklearn.log_model(pipeline, name="linear_regression_model")
    # mlflow.sklearn.save_model(pipeline, "linear_regression_model_v1")
    print("Model training completed.")
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Parameters:
    model (Pipeline): The trained model.
    X_test (DataFrame): Test features.
    y_test (Series): Test target variable.
    
    Returns:
    float: The mean squared error of the model on the test set.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return mse
    
if __name__ == "__main__":
    entry_point()


