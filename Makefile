# =========================
# Project Configuration
# =========================
PYTHON := python
PIP := pip
MLFLOW_PORT := 5000

# =========================
# Environment Setup
# =========================
.PHONY: setup
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# =========================
# DVC Commands
# =========================
.PHONY: dvc-init
dvc-init:
	dvc init

.PHONY: dvc-pull
dvc-pull:
	dvc pull

.PHONY: dvc-push
dvc-push:
	dvc push

.PHONY: repro
repro:
	dvc repro

.PHONY: status
status:
	dvc status

# =========================
# Pipeline Stages (Manual)
# =========================
.PHONY: preprocess
preprocess:
	$(PYTHON) src/preprocess.py

.PHONY: train
train:
	$(PYTHON) src/train.py

.PHONY: evaluate
evaluate:
	$(PYTHON) src/evaluate.py

# =========================
# MLflow
# =========================
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --port $(MLFLOW_PORT)

# =========================
# Cleanup
# =========================
.PHONY: clean
clean:
	rm -rf models/*
	rm -rf mlruns
	rm -rf __pycache__
	rm -rf src/__pycache__
