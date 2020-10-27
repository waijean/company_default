import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)

from imblearn.pipeline import make_pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

from conf.tables import TRAIN_RAW_SET, TRAIN_RATIO_SET, TRAIN_COMBINED_SET
from modeling.utils import model_pipeline_io

RANDOM_STATE = 0


def run(train_set_name: list):
    train_data, target = model_pipeline_io.get_training_set(train_set_name)

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, target, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = make_pipeline(
        QuantileTransformer(),
        BalancedRandomForestClassifier(random_state=RANDOM_STATE),
    )

    # Evaluate performance on cross validation set
    cv_metrics = cross_validate(
        pipeline, X_train, y_train, cv=5, scoring=["precision", "recall", "f1"]
    )
    cv_metrics_df = pd.DataFrame(cv_metrics)
    average_cv_metrics = cv_metrics_df[
        ["test_precision", "test_recall", "test_f1"]
    ].mean()
    print(f"Metrics on CV set: \n{average_cv_metrics.to_string()}")

    # Evaluate performance on test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    (
        test_precision,
        test_recall,
        test_f1_score,
        test_support,
    ) = precision_recall_fscore_support(y_test, y_pred, beta=1.0)
    test_metrics = {
        "precision": test_precision,
        "recall": test_recall,
        "f1_score": test_f1_score,
    }
    print(f"Metrics on test set: \n{classification_report(y_test, y_pred)}\n")

    # Get feature importance
    feature_importance = pd.Series(
        data=pipeline[-1].feature_importances_, index=train_data.columns,
    ).sort_values(ascending=False)
    print(f"Top 10 features: \n{feature_importance[:10].to_string()}\n")

    return cv_metrics_df, test_metrics, feature_importance


if __name__ == "__main__":
    train_set_name = [TRAIN_COMBINED_SET]
    print(f"Backtesting on {train_set_name} dataset...")
    cv_metrics_df, test_metrics, feature_importance = run(train_set_name)
