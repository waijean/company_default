import logging.config
import os
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np

import mlflow
from mlflow.exceptions import MlflowException
from plotly.graph_objs._figure import Figure
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from plotly import express as px

from model_utils.constants import (
    LOG_CONFIG_PATH,
    X_COL,
    Y_COL,
    PIPELINE_HTML,
    SCORES_CSV,
    FEATURE_IMPORTANCE_PLOT,
    FEATURE_IMPORTANCE_CSV,
    RUN_ID,
)

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str, artifact_location: str):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location
        )
    except MlflowException:
        experiment_id = mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id


def set_tags(X_col: List[str], y_col: str, active_run):
    logger.info("Setting tags")
    mlflow.set_tag(X_COL, X_col)
    mlflow.set_tag(Y_COL, y_col)
    mlflow.set_tag(RUN_ID, active_run.info.run_id)


def get_params(pipeline: Pipeline):
    logger.info("Getting params from pipeline")
    full_params = pipeline.get_params(deep=True)
    params = {
        key: value
        for key, value in full_params.items()
        if key not in ["memory", "verbose", "steps"]
    }
    return params


def log_params(params: Dict[str, Any]):
    logger.info("Logging params")
    mlflow.log_params(params)


def log_pipeline(pipeline: Pipeline):
    logger.info("Logging pipeline artifact")
    with open(PIPELINE_HTML, "w") as f:
        f.write(estimator_html_repr(pipeline))
    mlflow.log_artifact(PIPELINE_HTML)
    os.remove(PIPELINE_HTML)


def log_metrics(metrics: Dict[str, Any]):
    logger.info("Logging metrics")
    mlflow.log_metrics(metrics)


def log_cv_metrics(cv_results: Dict[str, Any]):
    """
    1. Log the average scores/fit time/score time across all folds as metric
    2. Convert original cv results to dataframe and log the dataframe
    """
    # remove estimator from cv_results dictionary to log metric and dataframe
    cv_results_without_estimator = {
        key: array for key, array in cv_results.items() if key != "estimator"
    }
    for key, array in cv_results_without_estimator.items():
        mlflow.log_metric(key, np.mean(array))
    log_df_artifact(pd.DataFrame(cv_results_without_estimator), SCORES_CSV)


def log_df_artifact(df: Union[pd.DataFrame, pd.Series], filename: str):
    df.to_csv(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)


def log_explainability(fitted_classifier, X_train):
    """
    Disable logging plotly artifact to save memory
    """
    logger.info("Logging explainability")
    if hasattr(fitted_classifier, "feature_importances_"):
        feature_importance = pd.Series(
            data=fitted_classifier.feature_importances_, index=X_train.columns,
        ).sort_values()
        log_df_artifact(
            feature_importance.sort_values(ascending=False), FEATURE_IMPORTANCE_CSV
        )
        # feature_importance_fig = px.bar(
        #     feature_importance,
        #     x=feature_importance.values,
        #     y=feature_importance.index,
        #     orientation="h",
        #     title="Feature Importance Plot",
        # )
        # log_plotly_artifact(feature_importance_fig, FEATURE_IMPORTANCE_PLOT)


def log_plotly_artifact(fig: Figure, filename: str):
    fig.write_html(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)
