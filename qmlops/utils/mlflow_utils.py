# src/utils/mlflow_utils.py
import mlflow
from contextlib import contextmanager

@contextmanager
def start_run(experiment_name, tags=None):
    """Start an MLflow run under the given experiment name."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        if tags:
            mlflow.set_tags(tags)
        yield run

def log_params(params: dict):
    mlflow.log_params(params)

def log_metrics(metrics: dict, step=None):
    mlflow.log_metrics(metrics, step=step)

def set_tags(tags: dict):
    mlflow.set_tags(tags)
