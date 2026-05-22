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
* **Result:** Evidently successfully detected significant dataset drift. 
* **Action:** The monitoring script was configured with a 30% drift threshold. Because the simulated drift exceeded this limit, the pipeline correctly triggered a failure exit code (sys.exit(1)), preventing the ingestion of anomalous data.