"""
Pure-Python logic for the LLM interface: feature parsing, model inference,
and response generation.  No Streamlit imports — safe to import in tests.
"""
import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NEBIUS_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Client is built lazily so the module can be imported without a key (e.g. in tests).
_client: "OpenAI | None" = None


def client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NEBIUS_API_KEY environment variable is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        _client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key,
        )
    return _client

FEATURE_SCHEMA = {
    "age":      "int, patient age in years (0–120)",
    "sex":      "int, 1=male 0=female",
    "cp":       "int, chest pain type: 1=typical angina 2=atypical angina 3=non-anginal 4=asymptomatic",
    "trestbps": "int, resting blood pressure in mmHg",
    "chol":     "int, serum cholesterol in mg/dL",
    "fbs":      "int, fasting blood sugar >120 mg/dL: 1=yes 0=no",
    "restecg":  "int, resting ECG: 0=normal 1=ST-T abnormality 2=LV hypertrophy",
    "thalach":  "int, maximum heart rate achieved",
    "exang":    "int, exercise-induced angina: 1=yes 0=no",
    "oldpeak":  "float, ST depression induced by exercise relative to rest",
    "slope":    "int, slope of peak-exercise ST segment: 1=upsloping 2=flat 3=downsloping",
    "ca":       "int, number of major vessels coloured by fluoroscopy (0–3)",
    "thal":     "int, thalassemia: 3=normal 6=fixed defect 7=reversible defect",
}

REQUIRED_FEATURES = list(FEATURE_SCHEMA.keys())


def parse_features(user_input: str) -> dict:
    """
    Call Nebius AI to extract feature values from a natural-language description.
    Returns a dict where missing/ambiguous values are None.
    """
    schema_lines = "\n".join(f"  - {k}: {v}" for k, v in FEATURE_SCHEMA.items())
    prompt = f"""Extract heart-disease risk-model features from this patient description.

Features to extract:
{schema_lines}

Patient message: "{user_input}"

Rules:
- Return ONLY valid JSON with exactly these 13 keys.
- Use null for any feature not mentioned or unclear.
- Convert natural language to numbers (e.g. "male"→1, "female"→0, "yes"→1, "no"→0).

JSON:"""

    response = client().chat.completions.create(
        model=NEBIUS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    raw_text = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if "```" in raw_text:
        raw_text = raw_text.split("```")[1].removeprefix("json").strip()

    return json.loads(raw_text)


def prepare_input(raw_features: dict) -> pd.DataFrame:
    """
    Convert a dict of raw feature values into a one-hot-encoded DataFrame row
    that matches the column structure used during training.
    """
    row = pd.DataFrame([{k: raw_features.get(k) for k in REQUIRED_FEATURES}])

    # thal was stored as float in training data (originally read from "3.0", "6.0", "7.0")
    # so we must cast here to get matching column names after get_dummies
    if "thal" in row.columns:
        row["thal"] = row["thal"].astype(float)

    categorical_cols = ["cp", "restecg", "slope", "thal"]
    cols_to_encode = [c for c in categorical_cols if c in row.columns]
    if cols_to_encode:
        row = pd.get_dummies(row, columns=cols_to_encode, drop_first=True)

    return row


def align_columns(user_df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    """Reindex user row to exactly match the columns the model was trained on."""
    return user_df.reindex(columns=model_columns, fill_value=0)


def load_best_model():
    """Load the highest-accuracy model artifact from MLflow."""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mlflow.set_tracking_uri(f"file:{os.path.join(project_root, 'mlruns')}")

    runs = mlflow.search_runs(
        experiment_names=["Heart_Disease_Pipeline_Prod"],
        order_by=["metrics.accuracy DESC"],
    )
    if runs.empty:
        raise RuntimeError("No MLflow runs found. Run src/train.py first.")

    best = runs.iloc[0]
    model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    return model, best


def generate_explanation(features: dict, prediction: int, probability: float) -> str:
    """Use Nebius AI to produce a plain-English explanation of the prediction."""
    risk = "HIGH" if prediction == 1 else "LOW"
    prompt = f"""You are a compassionate medical AI assistant. A patient used a heart-disease
risk-assessment tool and received the following result.

Patient features:
{json.dumps(features, indent=2)}

Model result: {risk} risk — {probability * 100:.1f}% probability of heart disease.

Write a warm, clear response (under 200 words) that:
1. States the risk level and probability plainly.
2. Highlights 2–3 features from the data that commonly drive this type of outcome.
3. Reminds the patient that this is an ML estimate, not a medical diagnosis.
4. Encourages them to discuss the results with a healthcare professional.

Response:"""

    response = client().chat.completions.create(
        model=NEBIUS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=350,
    )
    return response.choices[0].message.content.strip()
