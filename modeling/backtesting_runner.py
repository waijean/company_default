import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    fbeta_score,
)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC

from conf.tables import TRAIN_RAW_SET, TRAIN_RATIO_SET, TRAIN_COMBINED_SET
from modeling.utils import model_pipeline_io

RANDOM_STATE = 0


def get_cv_metrics(pipeline, X_train, y_train):
    # Evaluate performance on cross validation set
    cv_metrics = cross_validate(
        pipeline, X_train, y_train, cv=5, scoring=["precision", "recall", "f1"]
    )
    cv_metrics_df = pd.DataFrame(cv_metrics)
    average_cv_metrics = cv_metrics_df[
        ["test_precision", "test_recall", "test_f1"]
    ].mean()
    print(f"Metrics on CV set: \n{average_cv_metrics.to_string()}\n")
    return cv_metrics_df


def get_test_metrics(pipeline, X_test, y_test):
    # Evaluate performance on test set
    y_pred = pipeline.predict(X_test)
    test_metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": fbeta_score(y_test, y_pred, beta=1.0),
    }
    print(f"Metrics on test set: \n{classification_report(y_test, y_pred)}\n")
    return test_metrics


def get_feature_importance(pipeline, X):
    # Get feature importance
    feature_importance = pd.Series(
        data=pipeline[-1].feature_importances_, index=X.columns,
    ).sort_values(ascending=False)
    print(f"Top 10 features: \n{feature_importance[:10].to_string()}\n")
    return feature_importance


def run(X, y, test: bool, feature_importance: bool):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = make_pipeline(
        QuantileTransformer(),
        RandomUnderSampler(random_state=RANDOM_STATE),
        GradientBoostingClassifier(random_state=RANDOM_STATE),
    )

    cv_metrics_df = get_cv_metrics(pipeline, X_train, y_train)

    pipeline.fit(X_train, y_train)

    if feature_importance:
        feature_importance = get_feature_importance(pipeline, X)

    if test:
        test_metrics = get_test_metrics(pipeline, X_test, y_test)
        return cv_metrics_df, feature_importance, test_metrics

    return cv_metrics_df, feature_importance


if __name__ == "__main__":
    train_set_name = [TRAIN_COMBINED_SET]
    print(f"Backtesting on {train_set_name} dataset...")
    train_data, target = model_pipeline_io.get_training_set(train_set_name)
    result = run(train_data, target, test=True, feature_importance=True)
