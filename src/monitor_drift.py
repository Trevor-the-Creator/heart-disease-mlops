import os
import sys
import yaml
from preprocess import load_and_clean_data
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 1. Load the original reference data
with open("configs/params.yaml", "r") as file:
    config = yaml.safe_load(file)

data_path = config["data"]["raw_path"]
reference_data = load_and_clean_data(data_path)

# 2. Simulate Production Data Drift
# We copy the reference data and artificially inflate Age and Cholesterol 
# to simulate an older, higher-risk patient demographic arriving in production.
production_data = reference_data.copy()
production_data['age'] = production_data['age'] + 15  
production_data['chol'] = production_data['chol'] * 1.5

# 3. Initialize and run Evidently Data Drift Report
print("Running Evidently Data Drift analysis...")
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=production_data)

# 4. Save the interactive HTML report
os.makedirs("reports", exist_ok=True)
report_path = "reports/data_drift_report.html"
drift_report.save_html(report_path)
print(f"Interactive drift report saved to: {report_path}")

# 5. Extract results for the terminal summary
drift_results = drift_report.as_dict()
metrics_result = drift_results["metrics"][0]["result"]

drift_share = metrics_result["drift_share"]
drifted_columns = metrics_result["drift_by_columns"]

print("\n--- Drift Summary ---")
print(f"Overall Dataset Drift Share: {drift_share:.2%}")

print("\nDrifted Features:")
for feature, stats in drifted_columns.items():
    if stats["drift_detected"]:
        print(f" - {feature}")

# 6. Exit with code 1 if drift exceeds our 30% threshold
DRIFT_THRESHOLD = 0.30 

if drift_share > DRIFT_THRESHOLD:
    print(f"\nPipeline Failed: Drift share ({drift_share:.2%}) exceeds threshold of {DRIFT_THRESHOLD:.2%}.")
    sys.exit(1)
else:
    print("\nPipeline Passed: Data drift is within acceptable limits.")