# streamlit_app.py
import streamlit as st
from src.inference import load_pipeline, load_explainer, predict, explain

# Load artifacts once
pipeline = load_pipeline("artifacts/diabetes_pipeline.pkl")
explainer = load_explainer("artifacts/shap_explainer.pkl")

st.title("Early Diabetes Detection")

# Sidebar for user inputs
age = st.sidebar.slider("Age", 18, 120, 50)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
# …add other features…

input_df = { "age": age, "bmi": bmi, /* … */ }
df = st.experimental_data_editor(input_df, num_rows=1)

if st.button("Predict"):
    prob = predict(pipeline, df)
    st.metric("Diabetes Risk Probability", f"{prob:.2%}")
    shap_fig = explain(explainer, pipeline, df)
    st.pyplot(shap_fig)
