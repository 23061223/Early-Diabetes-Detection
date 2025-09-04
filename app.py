# app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Page config & title
st.set_page_config(
    page_title="Early Diabetes Risk Assessment",
    layout="centered"
)
st.title("🩺 Early Diabetes Risk Assessment")
st.markdown(
    "**Disclaimer:** This is a master’s project. "
    "The results are for reference only and not medical advice."
)
st.markdown("---")

# 2. Define paths
ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2023.csv"
PIPE_PATH = ROOT / "artifacts" / "diabetes_pipeline.pkl"
EXPL_PATH = ROOT / "artifacts" / "shap_explainer.pkl"

# 3. Load template and compute dummy-column set
df_template = pd.read_csv(DATA_PATH)
X_full      = df_template.drop(columns=["Diabetes_binary"])
X_cols      = pd.get_dummies(X_full, drop_first=True).columns

# 4. Cache and load pipeline & explainer
@st.cache_resource
def load_resources():
    pipe      = joblib.load(PIPE_PATH)
    explainer = joblib.load(EXPL_PATH)
    return pipe, explainer

pipe, explainer = load_resources()

# 5. Build Patient Information form
st.header("Patient Information")

weight = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    step=0.1
)
height = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    step=0.1
)

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

high_bp       = st.checkbox(
    "High Blood Pressure",
    help="High BP is typically systolic ≥130 mmHg or diastolic ≥80 mmHg."
)
high_chol     = st.checkbox(
    "High Cholesterol",
    help="High cholesterol often means LDL ≥130 mg/dL or total cholesterol ≥200 mg/dL."
)
age           = st.number_input(
    "Age (years)",
    min_value=18,
    max_value=120,
    step=1,
    help="Enter your actual age to assign an age group."
)
diff_walk     = st.checkbox("Serious difficulty walking or climbing stairs")
heart_disease = st.checkbox("History of heart disease or heart attack")
phys8         = st.checkbox(
    "Spent more than 8 hours in physical activity (excluding job) in last 30 days"
)
alcohol       = st.checkbox(
    "Heavy alcohol consumption",
    help="Men >14 drinks/week or women >7 drinks/week."
)

st.markdown("---")

# 6. Build and align feature vector
group = min((age - 18) // 5 + 1, 13)
features = {
    "BMI":                  bmi if bmi is not None else 0,
    "HighBP":               int(high_bp),
    "HighChol":             int(high_chol),
    "AgeGroup":             int(group),
    "DiffWalk":             int(diff_walk),
    "HeartDiseaseorAttack": int(heart_disease),
    "PhysHlth":             int(phys8),
    "HvyAlcoholConsump":    int(alcohol),
}
input_df = pd.DataFrame([features])
input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df.reindex(columns=X_cols, fill_value=0)

# 7. Predict & display results
if st.button("Assess Risk"):
    proba = pipe.predict_proba(input_df)[0, 1]

    # Color-coded risk score
    if proba < 0.5:
        st.success(f"Diabetes Risk Score: {proba:.2%}")
        st.write("🎉 Congrats! You have a low risk. Continue your healthy lifestyle.")
    else:
        st.error(f"Diabetes Risk Score: {proba:.2%}")
        st.write("⚠️ High risk. Consider a medical checkup and maintain a healthy lifestyle.")

    # 8. SHAP interpretability with unified API
    X_scaled  = pipe.named_steps["scaler"].transform(input_df)
    shap_exp  = explainer(X_scaled)

    st.subheader("Feature Contributions")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_exp[0], max_display=8, show=False)
    st.pyplot(fig)

st.markdown("---")
st.markdown("**Disclaimer:** This assessment is for reference only, not a clinical diagnosis.")
