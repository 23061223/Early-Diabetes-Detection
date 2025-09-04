import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Page config & title
st.set_page_config(page_title="Early Diabetes Risk Assessment", layout="centered")
st.title("🩺 Early Diabetes Risk Assessment")
st.markdown(
    "**Disclaimer:** This is a master’s project. "
    "The results are for reference only and not medical advice."
)
st.markdown("---")

# 2. Define paths
ROOT      = Path(__file__).parent
PIPE_PATH = ROOT / "artifacts" / "diabetes_pipeline.pkl"
EXPL_PATH = ROOT / "artifacts" / "shap_explainer.pkl"
COLS_PATH = ROOT / "artifacts" / "feature_columns.pkl"

# 3. Cache and load pipeline, explainer, and feature-column list
@st.cache_resource
def load_resources():
    pipe       = joblib.load(PIPE_PATH)
    explainer  = joblib.load(EXPL_PATH)
    feature_cols = joblib.load(COLS_PATH)
    return pipe, explainer, feature_cols

pipe, explainer, X_cols = load_resources()

# 4. Patient Information form
st.header("Patient Information")

weight = st.number_input("Weight (kg)", 30.0, 200.0, step=0.1)
height = st.number_input("Height (cm)", 100.0, 250.0, step=0.1)

bmi = None
if weight and height:
    bmi = weight / ((height / 100) ** 2)
    st.write(f"**Calculated BMI:** {bmi:.1f}")
    if bmi < 18.5:
        st.warning("Underweight (BMI < 18.5). Consider a health checkup.")
    elif bmi < 25:
        st.success("Healthy BMI (18.5–24.9). Keep it up!")
    elif bmi < 30:
        st.info("Overweight (BMI 25–29.9). Maintain a balanced diet and exercise.")
    else:
        st.error("Obese (BMI ≥ 30). Please consult a healthcare professional.")

high_bp       = st.checkbox("High Blood Pressure",
                            help="Typically systolic ≥130 or diastolic ≥80 mmHg.")
high_chol     = st.checkbox("High Cholesterol",
                            help="LDL ≥130 mg/dL or total ≥200 mg/dL.")
age           = st.number_input("Age (years)", 18, 120, step=1)
diff_walk     = st.checkbox("Serious difficulty walking or climbing stairs")
heart_disease = st.checkbox("History of heart disease or heart attack")
phys8         = st.checkbox(
    " >8 hours of recreational physical activity in last 30 days")
alcohol       = st.checkbox("Heavy alcohol consumption",
                            help="Men >14 drinks/week or women >7 drinks/week.")

st.markdown("---")

# 5. Build feature vector and align with training columns
group = min((age - 18) // 5 + 1, 13)
features = {
    "BMI":                  bmi or 0,
    "HighBP":               int(high_bp),
    "HighChol":             int(high_chol),
    "AgeGroup":             int(group),
    "DiffWalk":             int(diff_walk),
    "HeartDiseaseorAttack": int(heart_disease),
    "PhysHlth":             int(phys8),
    "HvyAlcoholConsump":    int(alcohol),
}
input_df = pd.get_dummies(pd.DataFrame([features]), drop_first=True)
input_df = input_df.reindex(columns=X_cols, fill_value=0)

# 6. Prediction & SHAP explanation
if st.button("Assess Risk"):
    proba = pipe.predict_proba(input_df)[0, 1]

    if proba < 0.5:
        st.success(f"Diabetes Risk Score: {proba:.2%}")
        st.write("🎉 Low risk—keep up the healthy habits!")
    else:
        st.error(f"Diabetes Risk Score: {proba:.2%}")
        st.write("⚠️ High risk—consider a medical checkup.")

    # Unified SHAP API
    X_scaled = pipe.named_steps["scaler"].transform(input_df)
    shap_exp = explainer(X_scaled)

    st.subheader("Feature Contributions")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_exp[0], max_display=8, show=False)
    st.pyplot(fig)

st.markdown("---")
st.markdown("**Disclaimer:** For reference only, not a clinical diagnosis.")
