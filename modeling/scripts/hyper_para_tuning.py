import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from imblearn.under_sampling import RandomUnderSampler

from model_utils import model_pipeline_io

RANDOM_STATE = 0


def define_random_grid() -> dict:
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=20)]
    max_features = ["auto", "sqrt"]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 8]
    bootstrap = [True, False]

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    return random_grid


def get_tuned_hyperpara(X: pd.DataFrame, y: pd.DataFrame):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    random_grid = define_random_grid()
    rf_random = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=random_grid,
        n_iter=100,
        cv=5,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )
    rf_random.fit(X_res, y_res)
    return rf_random


def get_training_set(train_set_name: list):
    train_data = pd.concat(
        [
            model_pipeline_io.read_train_file(name).iloc[:, :-1]
            for name in train_set_name
        ],
        axis=1,
    )
    target = model_pipeline_io.read_train_file(train_set_name[0]).iloc[:, -1]
    return train_data, target


def run(train_set_name: list):
    train_data, target = get_training_set(train_set_name)

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, target, test_size=0.2, random_state=0
    )

    rf_random = get_tuned_hyperpara(X_train, y_train)
    print(f"best parameters are : {rf_random.best_params_}")

    score = rf_random.best_estimator_.score(X_test, y_test)
    print(f"best estimator performance on test set: {score}")


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
