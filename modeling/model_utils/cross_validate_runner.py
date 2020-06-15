from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import mlflow
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
from sklearn.preprocessing import OneHotEncoder

from model_utils.mlrun import (
    set_tags,
    log_params,
    log_pipeline,
    log_cv_metrics,
    log_explainability,
    setup_mlflow,
    get_params,
)
from model_utils.scoring import BINARY_CLASSIFIER_SCORING
from model_utils.constants import (
    TRACKING_URI_PATH,
    ARTIFACT_PATH,
    DEFAULT_CV,
)
from model_utils.modeling import evaluate_cv_pipeline, load_data, split_data
from utils.constants import BANKRUPTCY_LABEL, BALANCE_SHEET, INCOME_STATEMENT


@dataclass
class CrossValidatePipeline:
    """
    Create a cross- validation pipeline to train and evaluate a specific pipeline

    Args:
        experiment_name: the experiment name for the grouping of runs
        read_path: path to parquet file which contains both feature and target columns
        X_col: list of feature columns
        y_col: target column
        params: parameters to log
        pipeline: sklearn Pipeline which can contain a series of estimator. It must have a classifier for the last step
        tracking_uri: location to store run details such as params, tags and metrics
        artifact_location: location to store artifacts such as html representation of pipeline, plots and models
    """

    experiment_name: str
    run_name: str
    read_path: str
    X_col: List
    y_col: str
    pipeline: Pipeline
    scoring: Dict
    is_column_trans: bool = False
    params: Optional[Dict[str, Any]] = None
    cv = DEFAULT_CV
    tracking_uri: str = TRACKING_URI_PATH
    artifact_location: Optional[str] = None

    def main(self):
        experiment_id = setup_mlflow(
            self.tracking_uri, self.experiment_name, self.artifact_location
        )
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=self.run_name
        ) as active_run:
            X, y = load_data(self.read_path, self.X_col, self.y_col)
            X_train, X_test, y_train, y_test = split_data(X, y)
            set_tags(self.X_col, self.y_col, active_run)
            if self.params is None:
                self.params = get_params(self.pipeline)
            log_params(self.params)
            log_pipeline(self.pipeline)
            column_trans, fitted_classifier, cv_results = evaluate_cv_pipeline(
                self.pipeline,
                X_train,
                y_train,
                self.scoring,
                self.cv,
                self.is_column_trans,
            )
            log_cv_metrics(cv_results)
            log_explainability(fitted_classifier, X_train, column_trans)


if __name__ == "__main__":
    # create pipeline
    column_trans = ColumnTransformer(
        [("CLUSTER_LABEL", OneHotEncoder(), ["CLUSTER_LABEL"]),],
        remainder="passthrough",
    )

    pipeline = make_pipeline_with_sampler(
        column_trans,
        SimpleImputer(strategy="constant", fill_value=0),
        RandomUnderSampler(random_state=42),
        BalancedRandomForestClassifier(random_state=42),
    )

    CrossValidatePipeline(
        experiment_name="MVP",
        run_name="Balanced Random Forest",
        read_path="D:/dev/Project/company_default/data/output/cleaned_raw_cluster_train.csv",
        X_col=INCOME_STATEMENT + BALANCE_SHEET + ["CLUSTER_LABEL"],
        y_col=BANKRUPTCY_LABEL,
        pipeline=pipeline,
        scoring=BINARY_CLASSIFIER_SCORING,
        is_column_trans=True,
    ).main()
