import mlflow
from mlflow.tracking import MlflowClient

# Set the experiment name we want to search
experiment_name = "Heart_Disease_Prediction"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"Could not find experiment '{experiment_name}'")
    exit()

# Query all runs in this experiment, sorted by accuracy (highest first)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"]
)

if runs.empty:
    print("No runs found. Make sure you trained the model first!")
else:
    # Extract the best run (the first row in our sorted dataframe)
    best_run = runs.iloc[0]
    
    print("--- Best Model Found ---")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Accuracy: {best_run['metrics.accuracy']:.3f}")
    print(f"Precision: {best_run['metrics.precision']:.3f}")
    print(f"Recall: {best_run['metrics.recall']:.3f}")
    print(f"Estimators: {best_run['params.n_estimators']}")
    print(f"Max Depth: {best_run['params.max_depth']}")