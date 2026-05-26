import sys
import yaml
from preprocess import load_and_clean_data, split_data
from evaluate import evaluate_model  # Importing our dedicated module
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# 1. Load the configuration file
with open("configs/params.yaml", "r") as file:
    config = yaml.safe_load(file)

data_path = config["data"]["raw_path"]
test_size = config["train"]["test_size"]
random_state = config["train"]["random_state"]
n_estimators = config["model"]["n_estimators"]
max_depth = config["model"]["max_depth"]

# 2. Process and split the data (Now includes One-Hot Categorical Encoding)
df = load_and_clean_data(data_path)
X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

# 3. Set up MLflow Tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_Pipeline_Prod")

with mlflow.start_run():
    # 4. Train the Model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # 5. Make Predictions
    predictions = model.predict(X_test)
    
    # Use decoupled evaluation function
    metrics = evaluate_model(y_test, predictions)
    accuracy = metrics["accuracy"]
    
    print(f"Model trained! Accuracy: {accuracy:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")

    # 6. Log everything to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    
    mlflow.set_tag("data_version", "encoded_cleveland_v2")
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print("Run logged to MLflow successfully!")

# 7. CI/CD Performance Threshold Check
if accuracy < 0.70:
    print(f"Pipeline Failed: Model accuracy ({accuracy:.3f}) is below the 0.70 threshold.")
    sys.exit(1)
else:
    print("Pipeline Passed: Model meets production thresholds.")