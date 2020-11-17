import numpy as np
import matplotlib.pyplot as plt
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer, fbeta_score

from sklearn.model_selection import learning_curve, train_test_split
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import QuantileTransformer

from conf.tables import TRAIN_COMBINED_SET
from modeling.utils import model_pipeline_io


def plot_learning_curve(estimator, X, y, cv, scoring):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    f, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, "o-", color="b", label="Validation score")
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="b",
    )
    plt.grid()
    plt.legend(loc="best")
    return ax


def plot_validation_curve(
    estimator, X, y, cv, scoring, param_name, param_range, log_scale=False
):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name, param_range, cv=cv, scoring=scoring
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    f, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Validation Curve")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Score")
    plt.plot(
        param_range,
        train_mean,
        color="r",
        marker="o",
        markersize=5,
        label="Training score",
    )
    plt.plot(
        param_range,
        test_mean,
        color="b",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation score",
    )
    plt.fill_between(
        param_range,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="r",
    )
    plt.fill_between(
        param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color="b"
    )
    plt.grid()
    plt.legend(loc="best")
    if log_scale:
        plt.xscale("log")

    return ax


def run(X, y, learning_curve=False, validation_curve=False):
    RANDOM_STATE = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = make_pipeline(
        SelectKBest(score_func=f_classif, k=10),
        QuantileTransformer(),
        RandomUnderSampler(random_state=RANDOM_STATE),
        GradientBoostingClassifier(random_state=RANDOM_STATE),
    )

    if learning_curve:
        ax = plot_learning_curve(
            pipeline, X_train, y_train, cv=5, scoring=make_scorer(fbeta_score, beta=2)
        )
        plt.show()

    if validation_curve:
        ax = plot_validation_curve(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=make_scorer(fbeta_score, beta=2),
            param_name="selectkbest__k",
            param_range=[10, 20, 30, 40, 50],
        )
        plt.show()


if __name__ == "__main__":
    train_set_name = [TRAIN_COMBINED_SET]
    print(f"Running diagnosis on {train_set_name} dataset...")
    train_data, target = model_pipeline_io.get_training_set(train_set_name)
    run(train_data, target, validation_curve=True)
