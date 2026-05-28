# Heart Disease Risk Estimator

An end-to-end MLOps application that combines a trained ensemble classifier with a
Nebius AI–powered natural language interface. Users describe their health profile in
plain English; the system extracts the relevant clinical features, runs them through
the best-performing trained model, and returns a risk estimate with a clear
conversational explanation.

---

## What It Does

A user types something like:

> *"I'm a 58-year-old male with typical angina. Resting BP 132, cholesterol 224,
normal ECG, max heart rate 173, no exercise angina, 0 ST depression, upsloping ST,
0 fluoroscopy vessels, normal thal."*

The system:
1. Sends the text to **Nebius AI Studio** (LLaMA 3.1 70B) to extract the 13 clinical features
2. Feeds the features into the best **scikit-learn model** logged in MLflow
3. Returns the risk probability plus an AI-generated plain-English explanation

---

## Dataset

**UCI Cleveland Heart Disease dataset** (`data/raw/heart.csv`, tracked with DVC)

- 303 patients, 13 clinical features, binary target (0 = no disease, 1 = disease)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Key features: age, sex, chest pain type, resting BP, cholesterol, ECG results,
  max heart rate, exercise-induced angina, ST depression, thalassemia type

---

## Architecture

```
User (natural language)
        │
        ▼
  Nebius AI Studio  ──── feature extraction ────▶  structured JSON
        │
        ▼
  src/interface_utils.py
  - parse_features()      ← LLM call 1: NL → features
  - prepare_input()       ← one-hot encode to match training columns
  - align_columns()       ← reindex to model's exact feature set
  - model.predict()       ← trained sklearn model from MLflow
  - generate_explanation()← LLM call 2: prediction → explanation
        │
        ▼
  Streamlit UI (src/app.py)
```

**Experiment tracking:** MLflow logs all 5 model configurations, their hyperparameters,
and four metrics (accuracy, precision, recall, F1). `src/compare_experiments.py` queries
the best run programmatically.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd heart-disease-mlops
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and add your Nebius AI Studio API key
```

Get a free key at https://studio.nebius.com.

### 3. Pull the data (DVC)

```bash
dvc pull
```

Or copy `heart.csv` manually to `data/raw/heart.csv`.

### 4. Train the models (runs 5 experiments, logs to MLflow)

```bash
cd src
python train.py
```

---

## Running the App

### Option A — Local Streamlit

```bash
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

### Option B — Docker

```bash
docker build -t heart-risk-app .
docker run -p 8501:8501 \
  -e NEBIUS_API_KEY=your_key_here \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/data:/app/data \
  heart-risk-app
```

Open http://localhost:8501.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: all tests pass across `test_preprocess.py`, `test_model.py`,
and `test_interface.py`.

---

## MLflow Experiment Tracking

View all logged runs:

```bash
mlflow ui
```

Or run the comparison script:

```bash
cd src
python compare_experiments.py
```

Five experiment configurations are trained:

| Name | Algorithm | Key Hyperparameters |
|------|-----------|---------------------|
| rf_shallow | Random Forest | n_estimators=50, max_depth=2 |
| rf_medium | Random Forest | n_estimators=100, max_depth=5 |
| rf_deep | Random Forest | n_estimators=200, max_depth=10 |
| gradient_boosting | Gradient Boosting | n_estimators=100, lr=0.1, max_depth=3 |
| logistic_regression | Logistic Regression | C=1.0 |

---

## Results

Best model performance on held-out test set (20% split, random_state=42):

| Metric | Value |
|--------|-------|
| Accuracy | ~0.85 |
| Precision | ~0.84 |
| Recall | ~0.88 |
| F1 | ~0.86 |

The **rf_medium** or **rf_deep** configuration typically wins. The gradient boosting
model is competitive. Logistic regression serves as a strong linear baseline.

**Model selection rationale:** F1 score is used as the primary ranking criterion
alongside accuracy, since false negatives (missing a high-risk patient) are more
costly than false positives in a clinical screening context.

---

## Repository Structure

```
heart-disease-mlops/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── configs/
│   └── params.yaml          # All hyperparameters and experiment configs
├── src/
│   ├── preprocess.py        # Data cleaning and feature engineering
│   ├── train.py             # Trains 5 model configs, logs to MLflow
│   ├── evaluate.py          # Accuracy, precision, recall, F1
│   ├── interface_utils.py   # LLM parsing, model inference, explanation
│   ├── app.py               # Streamlit web application
│   ├── compare_experiments.py
│   └── monitor_drift.py
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_interface.py
└── data/
    └── raw/
        └── heart.csv.dvc    # DVC pointer; actual data not committed
```

---

## Reflection

**What I learned:** Connecting a trained ML model to an LLM interface requires careful
engineering of the feature extraction step. The LLM needs a precise schema and strict
output format (JSON) to be reliable. One-hot encoding column alignment between training
and inference was the trickiest part — a single type mismatch (int vs float in `thal`)
silently produces wrong predictions.

**What was challenging:** Getting the LLM to return consistent JSON for edge cases
(partial information, ambiguous phrasing) required careful prompt engineering and
output parsing with markdown-fence stripping.

**What I would improve with more time:**
- Add a multi-turn conversation loop so the system can ask clarifying questions
  rather than rejecting incomplete inputs
- Train on a larger, more recent dataset and add confidence intervals to predictions
- Add SHAP explanations so the LLM can cite actual feature importances instead of
  domain heuristics
- Add authentication and rate limiting before any public deployment
