import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, precision_score

from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import RandomUnderSampler
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
        StandardScaler(),
        RandomUnderSampler(random_state=RANDOM_STATE),
        BalancedRandomForestClassifier(random_state=RANDOM_STATE),
    )

    # Evaluate performance on cross validation set
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    cv_accuracy = np.mean(cv_scores)
    print(f"Average accuracy on CV set: {cv_accuracy}")

    # Evaluate performance on test set
    pipeline.fit(X_train, y_train)
    test_accuracy = pipeline.score(X_test, y_test)
    y_pred = pipeline.predict(X_test)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    print(f"Accuracy on test set: {test_accuracy}")
    print(f"Recall on test set: {test_recall}")
    print(f"Precision on test set: {test_precision}")
    print(classification_report_imbalanced(y_test, y_pred))

    # Get feature importance
    feature_importance = pd.Series(
        data=pipeline[-1].feature_importances_, index=train_data.columns,
    ).sort_values(ascending=False)
    print(f"Top 10 important features: \n{feature_importance[:10]}")


if __name__ == "__main__":
    train_set_name = [TRAIN_RAW_SET]
    print(f"Modeling on raw dataset...")
    run(train_set_name)

    train_set_name = [TRAIN_RATIO_SET]
    print(f"Modeling on ratio dataset...")
    run(train_set_name)

    train_set_name = [TRAIN_COMBINED_SET]
    print(f"Modeling on combined dataset...")
    run(train_set_name)
