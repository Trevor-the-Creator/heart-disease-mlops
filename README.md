# Heart Disease Prediction: MLOps Pipeline

## Project Overview
This project demonstrates a complete MLOps pipeline for predicting heart disease using the UCI Cleveland dataset. It transitions a standard machine learning model into a production-ready system with automated testing, experiment tracking, and drift monitoring.

## Architecture
* **Configuration Management:** Hyperparameters are decoupled from code using `configs/params.yaml`.
* **Experiment Tracking:** MLflow logs all model parameters, metrics (Accuracy, Precision, Recall), and artifacts.
* **CI/CD Pipeline:** GitHub Actions automatically runs a 11-test Pytest suite and trains the model on every push.
* **Model Validation:** The pipeline hard-fails if model accuracy drops below 70%.

## Data Drift Analysis (Evidently)
To test the system's robustness, we simulated production data drift by inflating patient age by 15 years and cholesterol by 50%. 
## Data Drift Analysis (Evidently)
To test the system's robustness, we simulated production data drift.
* **Drifted Features:** `age` (inflated by 15 years) and `chol` (cholesterol, inflated by 50%).
* **Likely Impact on Model:** Because age and cholesterol are critical predictive features for heart disease, this data drift would likely force the model to over-predict risk, drastically increasing False Positives and degrading overall precision in production.
* **Recommended Action:** The monitoring script successfully detected this drift (>30% threshold) and halted the pipeline via a non-zero exit code. The data engineering team must investigate the upstream data source for extraction errors and quarantine the anomalous data before automated retraining can safely resume.