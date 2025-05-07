import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

# Set up MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Specify the MLflow tracking server URI
mlflow.set_experiment("Experience-to-Salary-Prediction")  # Set the name of the experiment in MLflow

# Sample data (you can replace this with real data from a CSV file)
data = {
    "Years of Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [5000, 6000, 7000, 8000, 10000, 11000, 13000, 14000, 15000, 16000]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[["Years of Experience"]]
y = df["Salary"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the model parameters
params = {"max_depth": 3, "random_state": 42}
model = RandomForestRegressor(**params)  # Initialize the Random Forest model with the specified parameters

# Start a new MLflow run to log the experiment
with mlflow.start_run(run_name="Experience-to-Salary-Prediction-Run") as run:
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error (MSE) between the predicted and actual salaries
    mse = mean_squared_error(y_test, y_pred)
    
    # Log the model parameters and metrics with MLflow
    mlflow.log_params(params)  # Log the parameters used to train the model
    mlflow.log_metrics({"mse": mse})  # Log the MSE as a metric
    
    # Log the trained model with MLflow, including the signature of the model (input-output relationship)
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(
        sk_model=model,  # The trained model
        artifact_path="sklearn-model",  # Path to store the model artifacts
        signature=signature,  # Input-output signature of the model
        registered_model_name="sk-learn-random-forest-reg-model"  # Name for the registered model
    )
