import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

from model_utils import model_pipeline_io

RANDOM_STATE = 42
TRAIN_RATIO_SET = "cleaned_ratio_train.csv"
TRAIN_RAW_SET = "cleaned_raw_train.csv"


def run(train_set_name: list):
    train_data, target = model_pipeline_io.get_training_set(train_set_name)

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, target, test_size=0.2, random_state=0
    )

    pipe = make_pipeline(
        StandardScaler(),
        SMOTE(random_state=RANDOM_STATE),
        BalancedRandomForestClassifier(random_state=RANDOM_STATE),
    )

    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    score = np.mean(scores)
    print(f"model average accuracy on train set: {score}")

    print("performance on test set...")
    pipe.fit(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"model accuracy on test set: {test_score}")
    print(classification_report_imbalanced(y_test, pipe.predict(X_test)))

    feature_importance = pd.Series(
        data=pipe[-1].feature_importances_, index=train_data.columns,
    ).sort_values(ascending=False)
    print(f"top 10 important features: \n{feature_importance[:10]}")


if __name__ == "__main__":
    train_set_name = [TRAIN_RAW_SET]
    print(f"try raw dataset...")
    run(train_set_name)

    train_set_name = [TRAIN_RATIO_SET]
    print(f"try ratio dataset...")
    run(train_set_name)

    train_set_name = [TRAIN_RAW_SET, TRAIN_RATIO_SET]
    print(f"try combine dataset...")
    run(train_set_name)
