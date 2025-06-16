import pandas as pd
import pickle
import logging
from ucimlrepo import fetch_ucirepo
from explainability.pipeline import ExplainabilityPipeline
from pathlib import Path
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

mlflow.set_tracking_uri("http://mlflow:5001")
experiment = mlflow.set_experiment("Explainability Monitoring")
print(f"Artifact location: {experiment.artifact_location}")

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

heart = fetch_ucirepo(id=45)
X = heart.data.features.fillna(heart.data.features.median())

config = {
    "use_shap": True,
    "use_lime": True
}

pipeline = ExplainabilityPipeline(config)
report = pipeline.run_explainability_check(model, X, model_version="heart_disease_v1")