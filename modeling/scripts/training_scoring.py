import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import RandomUnderSampler

from model_utils import model_pipeline_io

RANDOM_STATE = 0


def run(train_set_name: list):
    train_data, target = model_pipeline_io.get_training_set(train_set_name)

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, target, test_size=0.2, random_state=0
    )

    pipe = make_pipeline(
        StandardScaler(),
        RandomUnderSampler(random_state=RANDOM_STATE),
        RandomForestClassifier(random_state=RANDOM_STATE),
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
    train_set_name = ["cleaned_raw_train.csv"]
    print(f"try raw dataset...")
    run(train_set_name)

    train_set_name = ["cleaned_ratio_train.csv"]
    print(f"try ratio dataset...")
    run(train_set_name)

    train_set_name = ["cleaned_raw_train.csv", "cleaned_ratio_train.csv"]
    print(f"try combine dataset...")
    run(train_set_name)
