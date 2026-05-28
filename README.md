# Heart Disease Risk Estimator

An end-to-end MLOps application that combines a trained ensemble classifier with a
Nebius AI–powered natural language interface. Users describe their health profile in
plain English; the system extracts the relevant clinical features, runs them through
the best-performing trained model, and returns a risk estimate with a clear
conversational explanation.

**Who it is for:** Patients and clinicians who want a quick, conversational first-pass
risk screen before a formal medical consultation.  
**Problem it solves:** Traditional risk calculators require filling out rigid forms.
This application lets anyone describe their situation naturally and receive an
instant, explainable estimate.

---

## Dataset

**UCI Cleveland Heart Disease dataset** (`data/raw/heart.csv`, tracked with DVC)

- 303 patients, 13 clinical features, binary target (0 = no disease, 1 = disease present)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Features: age, sex, chest pain type (cp), resting BP (trestbps), cholesterol (chol),
  fasting blood sugar (fbs), resting ECG (restecg), max heart rate (thalach),
  exercise-induced angina (exang), ST depression (oldpeak), ST slope (slope),
  fluoroscopy vessels (ca), thalassemia type (thal)

---

## Data Preprocessing

The full pipeline lives in `src/preprocess.py`. Each step is documented below.

| Step | What happens | Why |
|------|-------------|-----|
| Load with `na_values=["?"]` | Converts the `?` missing-value sentinel to `NaN` | The raw Cleveland file uses `?` rather than blank cells |
| Auto-assign column names | Adds standard headers if the file ships without them | Some mirrors of the dataset omit the header row |
| `dropna()` | Removes 6 incomplete rows (all in `ca` or `thal`) | Avoids imputation bias on a small dataset |
| Binarize target | Maps 0 → 0, 1–4 → 1 | Converts the 5-class severity scale to a binary disease/no-disease label |
| One-hot encode `cp`, `restecg`, `slope`, `thal` | `pd.get_dummies(..., drop_first=True)` | These are nominal categoricals — integer encoding would impose a false ordinal relationship |
| `StandardScaler` on `NUMERIC_COLS` | Fit on train set only, transform both train and test | Prevents test-set statistics from leaking into the scaler; required for Logistic Regression to converge properly |

**Columns scaled:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`  
**Columns not scaled:** binary indicators (`sex`, `fbs`, `exang`) and all one-hot encoded columns — scaling them would not improve tree models and would distort LR coefficients.

The fitted scaler is saved as a `scaler.pkl` MLflow artifact alongside every model run, so the same transformation can be applied at inference time without recomputing statistics.

---

## Architecture

```
User (natural language)
        │
        ▼
  Nebius AI Studio (LLaMA 3.3 70B)
  ── parse_features() ──▶  structured JSON (13 features)
        │
        ▼
  src/interface_utils.py
  - prepare_input()        ← one-hot encode categoricals
  - align_columns()        ← reindex to model's exact column set
  - scale_for_inference()  ← apply fitted StandardScaler
  - model.predict()        ← best sklearn model loaded from MLflow
  - generate_explanation() ← LLM call 2: prediction → plain-English response
        │
        ▼
  Streamlit UI (src/app.py)
```

**Experiment tracking:** MLflow logs all 5 model configurations, their hyperparameters,
four metrics, and two artifacts (model + scaler) per run.
`src/compare_experiments.py` queries and ranks runs programmatically with
`mlflow.search_runs()`.

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
python src/train.py
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

## Demo

### Full prediction (all 13 features provided)

**Input:**
> I'm a 63-year-old male with typical angina. My resting blood pressure is 145,
cholesterol is 233, fasting blood sugar is above 120, ECG shows left ventricular
hypertrophy, max heart rate is 150, no exercise-induced angina, ST depression of 2.3,
downsloping ST slope, 0 fluoroscopy vessels, fixed defect thalassemia.

**Output:**
> 🔴 HIGH RISK — 87.3% probability  
> Based on the health information you provided, your model assessment indicates a high
> risk of heart disease. Key factors driving this result include your elevated resting
> blood pressure (145 mmHg), ST depression of 2.3, and typical angina — all strong
> clinical indicators. This is a machine learning estimate, not a medical diagnosis.
> Please discuss these results with your doctor.

### Edge case (incomplete input)

**Input:**
> I'm a 45-year-old woman with high cholesterol.

**Output (conversational clarification):**
> Thanks for sharing that! To run a full assessment I still need a few more details —
> things like your resting blood pressure, maximum heart rate during exercise, and
> whether you experience chest pain or shortness of breath during activity. Could you
> provide those?

---

## Running Tests

```bash
pytest tests/ -v
```

All 12 tests pass across `test_preprocess.py`, `test_model.py`, and `test_interface.py`.

---

## MLflow Experiment Tracking

```bash
mlflow ui          # visual dashboard at http://localhost:5000
python src/compare_experiments.py   # programmatic best-run query
```

---

## Results

All five models were evaluated on the same held-out 20% test split (random_state=42)
after StandardScaler preprocessing.

| Experiment | Algorithm | Accuracy | Precision | Recall | F1 |
|------------|-----------|----------|-----------|--------|----|
| rf_shallow | Random Forest (n=50, depth=2) | 0.867 | 0.833 | 0.833 | 0.833 |
| rf_medium | Random Forest (n=100, depth=5) | 0.867 | 0.808 | 0.875 | 0.840 |
| rf_deep | Random Forest (n=200, depth=10) | 0.850 | 0.800 | 0.833 | 0.816 |
| gradient_boosting | GradientBoosting (n=100, lr=0.1) | 0.833 | 0.750 | 0.875 | 0.808 |
| **logistic_regression** | **Logistic Regression (C=1.0)** | **0.883** | **0.870** | **0.833** | **0.851** |

**Best model: Logistic Regression** with accuracy 0.883 and F1 0.851.

**Selection rationale:** Logistic Regression achieves the highest accuracy and
precision. In a clinical screening context, false negatives (missing a high-risk
patient) are more costly than false positives, so F1 score is tracked alongside
accuracy. Logistic Regression also benefits most from the StandardScaler added in
this pipeline, and its coefficients are directly interpretable — a useful property
when explaining predictions to clinicians.

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
│   ├── preprocess.py        # Data cleaning, encoding, scaling (documented)
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
silently produces wrong predictions. Adding StandardScaler also required saving the
fitted scaler as an MLflow artifact so the exact same transformation could be applied
at inference time.

**What was challenging:** Getting the LLM to return consistent JSON for edge cases
(partial information, ambiguous phrasing) required careful prompt engineering and
output parsing with markdown-fence stripping. The conversational edge-case handler —
which generates a friendly clarifying question rather than a hard error — required a
second LLM call with a different prompt style.

**What I would improve with more time:**
- Add a multi-turn conversation loop so the system can ask clarifying questions
  iteratively rather than in one pass
- Train on a larger, more recent dataset and report confidence intervals alongside
  point estimates
- Add SHAP explanations so the LLM can cite actual feature importances instead of
  general domain heuristics
- Add authentication and rate limiting before any public deployment
