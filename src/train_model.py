import pandas as pd
import joblib
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

SEED = 42
DATA_PATH = "data/diabetes_binary_health_indicators_BRFSS2023.csv"
MODEL_PATH = "artifacts/diabetes_pipeline.pkl"
EXPLAINER_PATH = "artifacts/shap_explainer.pkl"

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"]
    return X, y

def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse=False))
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

def train_and_serialize():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=0.2,
        random_state=SEED
    )

    preprocessor = build_preprocessor(X_train)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", XGBClassifier(
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=SEED
        ))
    ])

    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__gamma": [0, 1, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=SEED,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_

    # Optional probability calibration
    from sklearn.calibration import CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(
        base_estimator=best_pipe,
        method="isotonic",
        cv=cv
    )
    calibrated.fit(X_train, y_train)

    # Serialize the final pipeline
    joblib.dump(calibrated, MODEL_PATH, compress=("gzip", 3))

    # Build and serialize SHAP explainer
    background = X_train.sample(n=100, random_state=SEED)
    explainer = shap.Explainer(
        lambda x: calibrated.predict_proba(x)[:, 1],
        background
    )
    joblib.dump(explainer, EXPLAINER_PATH, compress=("gzip", 3))

    print("Model and SHAP explainer saved.")

if __name__ == "__main__":
    train_and_serialize()
