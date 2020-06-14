import logging.config
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline

from model_utils.constants import LOG_CONFIG_PATH

logging.config.fileConfig(fname=LOG_CONFIG_PATH, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def load_data(read_path: str, X_col: List[str], y_col: str):
    logger.info(f"Loading data from {read_path}")
    X = pd.read_csv(read_path, usecols=X_col)
    y = pd.read_csv(read_path, usecols=[y_col])
    return X, y


def split_data(X: pd.DataFrame, y: pd.DataFrame):
    logger.info(f"Splitting data into train and test set")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test


def evaluate_cv_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.DataFrame, scoring: Dict, cv
):
    logger.info("Evaluating pipeline")
    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True,
    )
    # extract first pipeline from cv_results
    fitted_pipeline = cv_results["estimator"][0]
    # extract the last step from pipeline
    fitted_classifier = fitted_pipeline[-1]

    return fitted_classifier, cv_results


def evaluate_grid_search_pipeline(
    pipeline: Pipeline,
    param_grid,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    scoring: Dict,
    cv,
    refit: str,
):
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=refit,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs = gs.fit(X_train, y_train)
    best = {
        "params": gs.best_params_,
        "metrics": {refit: gs.best_score_},
        "fitted_classifier": gs.best_estimator_,
    }
    return best, gs.cv_results_
