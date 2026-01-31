# End-to-End Machine Learning Pipeline with DVC and MLflow

This project demonstrates how to build a **reproducible, experiment-driven machine learning pipeline** using **DVC** for data and model versioning and **MLflow** for experiment tracking.

The pipeline trains a **Random Forest Classifier** on the **Pima Indians Diabetes Dataset**, following clear and modular stages for preprocessing, training, and evaluation.

---

## Project Objectives

- Ensure reproducibility across environments  
- Enable experiment tracking and comparison  
- Promote clean collaboration in ML teams  
- Demonstrate practical MLOps fundamentals  

---

## Tech Stack

- Python  
- DVC (Data Version Control)  
- MLflow  
- Scikit-learn  
- Git / GitHub  

---

## Pipeline Overview

```text
Raw Data
   ↓
Preprocessing (DVC Stage)
   ↓
Processed Data
   ↓
Training (DVC + MLflow)
   ↓
Trained Model
   ↓
Evaluation (MLflow Metrics)
```

---

## Pipeline Stages

### 1. Preprocessing

- **Script:** `src/preprocess.py`
- **Input:** `data/raw/data.csv`
- **Output:** `data/processed/data.csv`
- Ensures consistent and repeatable data preparation

---

### 2. Training

- **Script:** `src/train.py`
- **Model:** Random Forest Classifier
- **Output:** `models/model.joblib`

**MLflow logs:**
- Hyperparameters  
- Model artifact  
- Accuracy metric  

---

### 3. Evaluation

- **Script:** `src/evaluate.py`
- Evaluates trained model  
- Logs metrics to MLflow for comparison  

---

## Project Structure

```text
.
├── data
│   ├── raw
│   │   └── data.csv
│   └── processed
│       └── data.csv
├── models
│   └── model.joblib
├── src
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── Makefile
├── README.md
```

---

## DVC Pipeline Definition

### Preprocessing Stage

```bash
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
```

---

### Training Stage

```bash
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/processed/data.csv \
    -o models/model.joblib \
    python src/train.py
```

---

### Evaluation Stage

```bash
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.joblib -d data/processed/data.csv \
    python src/evaluate.py
```

---

## How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/Ingle-Santosh/diabetes-prediction-dvc-mlflow-mlops.git
cd diabetes-prediction-dvc-mlflow-mlops
```

---

### 2. Install Dependencies

```bash
make setup
```

---

### 3. Pull Data & Models

```bash
make dvc-pull
```

---

### 4. Run Full Pipeline

```bash
make repro
```

---

## MLflow Experiment Tracking

### Start MLflow UI

```bash
make mlflow-ui
```

Open in browser:

```
http://localhost:5000
```

You can:
- Compare experiments  
- Inspect parameters and metrics  
- Download trained models  

---

## Why DVC + MLflow?

| Capability | DVC | MLflow |
|----------|-----|--------|
| Data versioning | ✅ | ❌ |
| Pipeline automation | ✅ | ❌ |
| Experiment tracking | ❌ | ✅ |
| Model artifacts | ❌ | ✅ |
| Collaboration | ✅ | ✅ |

Together, they cover the **entire ML lifecycle**.

---

## Use Cases

- Data science teams building reproducible pipelines  
- ML engineers operationalizing experiments  
- Researchers comparing multiple models and parameters  
