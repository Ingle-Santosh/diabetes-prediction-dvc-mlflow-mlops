import pandas as pd
import yaml
import logging
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import mlflow
from urllib.parse import urlparse

import joblib
from dotenv import load_dotenv
import os

# Environment & MLflow Setup
load_dotenv()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load Parameters
PARAMS_PATH = Path("params.yaml")

with PARAMS_PATH.open("r") as f:
    params = yaml.safe_load(f)["train"]

# Evaluation Function
def evaluate(data_path: str, model_path: str) -> None:
    data_path = Path(data_path)
    model_path = Path(model_path)

    logger.info("Loading dataset from %s", data_path)
    data = pd.read_csv(data_path)

    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    with mlflow.start_run():

        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)

        logger.info("Model Accuracy: %.4f", accuracy)

        # MLflow Logging
        mlflow.log_metric("accuracy", accuracy)

        cm = confusion_matrix(y, predictions)
        cr = classification_report(y, predictions)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        logger.info("Evaluation metrics logged to MLflow")


# Entry Point
if __name__ == "__main__":
    evaluate(
        data_path=params["data"],
        model_path=params["model"]
    )
