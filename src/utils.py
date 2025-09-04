from typing import Tuple
import pandas as pd
import joblib
from pydantic import BaseModel, Field

MODEL_PATH = "artifacts/diabetes_pipeline.pkl"
EXPLAINER_PATH = "artifacts/shap_explainer.pkl"

def load_artifacts() -> Tuple:
    pipe = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    return pipe, explainer

class Patient(BaseModel):
    BMI: float
    KidneyDisease: int = Field(ge=0, le=1)
    HighBP: int = Field(ge=0, le=1)
    HighChol: int = Field(ge=0, le=1)
    CholCheck: int = Field(ge=0, le=1)
    Asthma: int = Field(ge=0, le=1)
    COPD: int = Field(ge=0, le=1)
    Smoker: int = Field(ge=0, le=1)
    Stroke: int = Field(ge=0, le=1)
    HeartDiseaseorAttack: int = Field(ge=0, le=1)
    PhysActivity: int = Field(ge=0, le=1)
    HvyAlcoholConsump: int = Field(ge=0, le=1)
    AnyHealthcare: int = Field(ge=0, le=1)
    NoDocbcCost: int = Field(ge=0, le=1)
    GenHlth: int  = Field(ge=1, le=5)
    MentHlth: float
    PhysHlth: float
    DiffWalk: int = Field(ge=0, le=1)
    Sex: int      = Field(ge=0, le=1)
    AgeGroup: int = Field(ge=1, le=13)
    Education: int= Field(ge=1, le=6)
    Income: int   = Field(ge=1, le=8)
