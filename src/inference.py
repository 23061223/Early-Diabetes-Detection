from fastapi import FastAPI, HTTPException
import pandas as pd

from .utils import Patient, load_artifacts

app = FastAPI(title="Diabetes Risk API")

# load pipeline and explainer on startup
pipe, explainer = load_artifacts()

@app.post("/predict")
def predict(patient: Patient):
    df = pd.DataFrame([patient.dict()])
    try:
        proba = float(pipe.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"risk_score": round(proba, 4)}

@app.post("/explain")
def explain(patient: Patient):
    df = pd.DataFrame([patient.dict()])
    try:
        explanation = explainer(df)
        shap_vals = explanation.values[0]
        base_value = float(explanation.base_values[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    contributions = dict(zip(df.columns.tolist(), shap_vals.tolist()))
    return {"base_value": base_value, "contributions": contributions}
