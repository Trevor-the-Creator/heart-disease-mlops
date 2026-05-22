import sys
import yaml
from preprocess import load_and_clean_data, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

# 2. Process and split the data
df = load_and_clean_data(data_path)
X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)

# 3. Set up MLflow Tracking
# Force MLflow to use a local, relative folder
mlflow.set_tracking_uri("file:./mlruns")

# Change the name to force a completely fresh database
mlflow.set_experiment("Heart_Disease_Pipeline_Prod")

# Start the "flight data recorder"
with mlflow.start_run():
    
    # 4. Train the Model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # 5. Make Predictions & Evaluate
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    
    print(f"Model trained! Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

    # 6. Log everything to MLflow
    # Log the "dials and knobs"
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Log the evaluation results
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log a tag indicating what data we used
    mlflow.set_tag("data_version", "raw_cleveland_v1")
    
    # Log the actual model file so we can deploy it later
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print("Run logged to MLflow successfully!")

    # 7. CI/CD Performance Threshold Check
if accuracy < 0.70:
    print(f"Pipeline Failed: Model accuracy ({accuracy:.3f}) is below the 0.70 threshold.")
    sys.exit(1) # The non-zero exit code tells GitHub Actions to fail the job
else:
    print("Pipeline Passed: Model meets production thresholds.")