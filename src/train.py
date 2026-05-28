import sys
import pickle
import tempfile
import os
import yaml
from preprocess import load_and_clean_data, split_data, scale_features
from evaluate import evaluate_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

with open("configs/params.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["data"]["raw_path"]
test_size = config["train"]["test_size"]
random_state = config["train"]["random_state"]

df = load_and_clean_data(data_path)
X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
X_train, X_test, scaler = scale_features(X_train, X_test)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_Pipeline_Prod")

best_accuracy = 0.0
best_run_id = None

for exp in config.get("experiments", []):
    with mlflow.start_run(run_name=exp["name"]):
        model_type = exp["model_type"]

        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=exp["n_estimators"],
                max_depth=exp["max_depth"],
                random_state=random_state,
            )
            mlflow.log_param("n_estimators", exp["n_estimators"])
            mlflow.log_param("max_depth", exp["max_depth"])

        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=exp["n_estimators"],
                max_depth=exp["max_depth"],
                learning_rate=exp["learning_rate"],
                random_state=random_state,
            )
            mlflow.log_param("n_estimators", exp["n_estimators"])
            mlflow.log_param("max_depth", exp["max_depth"])
            mlflow.log_param("learning_rate", exp["learning_rate"])

        elif model_type == "logistic_regression":
            model = LogisticRegression(
                C=exp["C"],
                max_iter=exp["max_iter"],
                random_state=random_state,
            )
            mlflow.log_param("C", exp["C"])
            mlflow.log_param("max_iter", exp["max_iter"])

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_version", "encoded_cleveland_v2")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_model(y_test, predictions)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.set_tag("experiment_name", exp["name"])
        mlflow.sklearn.log_model(model, "model")

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path)

        print(
            f"[{exp['name']}] accuracy={metrics['accuracy']:.3f}  "
            f"f1={metrics['f1']:.3f}  precision={metrics['precision']:.3f}  "
            f"recall={metrics['recall']:.3f}"
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_run_id = mlflow.active_run().info.run_id

print(f"\nBest run: {best_run_id}  accuracy={best_accuracy:.3f}")

if best_accuracy < 0.70:
    print(f"Pipeline Failed: best accuracy ({best_accuracy:.3f}) below 0.70 threshold.")
    sys.exit(1)
else:
    print("Pipeline Passed: best model meets production thresholds.")
