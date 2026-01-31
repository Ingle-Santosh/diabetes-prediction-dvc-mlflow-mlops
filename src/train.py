import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse

import joblib
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_user = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")

if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load parameters
PARAMS_PATH = Path("params.yaml")

with PARAMS_PATH.open("r") as f:
    params = yaml.safe_load(f)["train"]



# Hyperparameter Tuning
def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, Any],
    random_state: int = 42
) -> GridSearchCV:

    rf = RandomForestClassifier(random_state=random_state)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    return grid_search


# Training Function
def train(
    data_path: str,
    model_path: str,
    random_state: int,
    n_estimators: int,
    max_depth: int
) -> None:

    data_path = Path(data_path)
    model_path = Path(model_path)

    logger.info("Loading dataset")
    data = pd.read_csv(data_path)

    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    with mlflow.start_run():

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)

        signature = infer_signature(X_train, y_train)

        # Hyperparameter grid
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        logger.info("Starting hyperparameter tuning")
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid, random_state)

        best_model = grid_search.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Accuracy: {accuracy:.4f}")

        # MLflow Logging
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_params(grid_search.best_params_)

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Log model
        tracking_url_type = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        if tracking_url_type != "file":
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name="Best Model"
            )
        else:
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                signature=signature
            )

        # Save model locally (joblib)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)

        logger.info(f"Model saved to {model_path}")


# script Entry Point
if __name__ == "__main__":
    train(
        data_path=params["data"],
        model_path=params["model"],
        random_state=params["random_state"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
    )
