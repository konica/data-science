import mlflow
from mlflow import sklearn
from sklearn.linear_model import LinearRegression

def inference():
    """
    Perform inference using the trained model.
    
    Returns:
    None
    """
    # Load the model
    model = mlflow.sklearn.load_model("./linear_regression_model_v1")
    # mlflow.set_tracking_uri("http://localhost:5000")
    # model = mlflow.sklearn.load_model( "runs:/b05b301d672045a5b5a6a639ec6204d9/linear_regression_model")
    
    # Example input for inference (replace with actual data)
    example_input = [[15.0, 1012, 3, 10.3]]  # Example features: ['min_temp', 'global_radiation', 'month', 'cloud_cover']
    
    # Perform prediction
    prediction = model.predict(example_input)
    
    print(f"Predicted mean temperature: {prediction[0]}°C")


if __name__ == "__main__":
    # Run inference
    inference()