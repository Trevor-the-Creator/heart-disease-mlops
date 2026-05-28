import sys
import os

# Ensure project root is on the path when running via `streamlit run src/app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from interface_utils import (
    parse_features,
    prepare_input,
    align_columns,
    scale_for_inference,
    load_best_model,
    generate_explanation,
    ask_for_missing_features,
    REQUIRED_FEATURES,
    FEATURE_SCHEMA,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Risk Estimator", page_icon="🫀")
st.title("🫀 Heart Disease Risk Estimator")
st.caption("Powered by a trained ML model + Nebius AI · Not a substitute for medical advice")

# ── Load model (cached so it only loads once) ────────────────────────────────
@st.cache_resource
def get_model():
    return load_best_model()

try:
    model, scaler, best_run = get_model()
    model_columns = (
        list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    )
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# ── How-to section ───────────────────────────────────────────────────────────
with st.expander("ℹ️ How to use this tool"):
    st.markdown("""
Describe your health profile in plain English. For example:

> *"I'm a 58-year-old male with typical angina. My resting BP is 132 mmHg,
cholesterol is 224 mg/dL, normal ECG, max heart rate 173, no exercise-induced
angina, 0 ST depression, upsloping ST, 0 fluoroscopy vessels, normal thal."*

The system will:
1. Extract your feature values using an LLM
2. Run them through the trained Random Forest / Gradient Boosting model
3. Return a risk estimate with a plain-English explanation

**Required features** (all 13 must be present for a prediction):
""")
    for k, v in FEATURE_SCHEMA.items():
        st.markdown(f"- **{k}**: {v}")

st.divider()

# ── Input ────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Describe your health profile:",
    placeholder=(
        "E.g. I'm a 55-year-old male, non-anginal chest pain, resting BP 130, "
        "cholesterol 250, no high fasting blood sugar, normal ECG, max heart rate 165, "
        "no exercise angina, 0 ST depression, flat ST slope, 0 vessels, normal thal."
    ),
    height=130,
)

if st.button("Assess Risk", type="primary"):
    if not user_input.strip():
        st.warning("Please describe your health profile above.")
        st.stop()

    # ── Step 1: parse features ────────────────────────────────────────────────
    with st.spinner("Parsing your health information…"):
        try:
            raw = parse_features(user_input)
        except Exception as e:
            st.error(f"Could not parse your input: {e}. Please try rephrasing.")
            st.stop()

    missing = [k for k in REQUIRED_FEATURES if raw.get(k) is None]

    if missing:
        with st.spinner("Generating follow-up question…"):
            try:
                clarification = ask_for_missing_features(raw, missing)
            except Exception:
                clarification = (
                    f"Thanks for sharing that! I still need a few more details to run "
                    f"the assessment: **{', '.join(missing)}**. Could you provide those?"
                )
        st.info(f"💬 {clarification}")
        with st.expander("What was extracted so far"):
            st.json({k: v for k, v in raw.items() if v is not None})
        st.stop()

    # ── Step 2: run model ─────────────────────────────────────────────────────
    with st.spinner("Running model…"):
        try:
            user_df = prepare_input({k: raw[k] for k in REQUIRED_FEATURES})
            if model_columns:
                user_df = align_columns(user_df, model_columns)
            user_df = scale_for_inference(user_df, scaler)

            prediction = int(model.predict(user_df)[0])
            probability = float(model.predict_proba(user_df)[0][1])
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # ── Step 3: generate explanation ──────────────────────────────────────────
    with st.spinner("Generating explanation…"):
        try:
            explanation = generate_explanation(raw, prediction, probability)
        except Exception as e:
            explanation = (
                f"Model prediction: {'HIGH' if prediction == 1 else 'LOW'} risk "
                f"({probability * 100:.1f}%). (Explanation unavailable: {e})"
            )

    # ── Display result ────────────────────────────────────────────────────────
    risk_icon = "🔴" if prediction == 1 else "🟢"
    risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
    st.markdown(f"### {risk_icon} {risk_label} — {probability * 100:.1f}% probability")
    st.markdown(explanation)

    with st.expander("Parsed feature values"):
        st.json(raw)

    with st.expander("Model details"):
        st.write(f"**Best model accuracy:** {best_run['metrics.accuracy']:.3f}")
        st.write(f"**Best model F1:** {best_run.get('metrics.f1', 'N/A')}")
        st.write(f"**Run ID:** {best_run['run_id']}")
