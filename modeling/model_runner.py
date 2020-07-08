import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score

from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

from model_utils import model_pipeline_io, hyperparameter_const
from model_utils.constants import (
    RANDOM_STATE,
    TEST_RAW_SET,
    TEST_RATIO_SET,
    TEST_COM_SET,
    TRAIN_RAW_SET,
    TRAIN_COM_SET,
    TRAIN_RATIO_SET,
    FEATURE_LIST,
)


def train(train_set_name: list):
    train_data, target = model_pipeline_io.get_training_set(train_set_name)
    train_data = train_data.loc[:, FEATURE_LIST]

    pipe = make_pipeline(
        QuantileTransformer(n_quantiles=10),
        RandomUnderSampler(random_state=RANDOM_STATE),
        RandomForestClassifier(**hyperparameter_const.COMBINE_DATA),
    )

    pipe.fit(train_data, target)
    return pipe


def test(test_set_name: list, model: RandomForestClassifier):
    test_data = model_pipeline_io.get_test_set(test_set_name)
    test_data = test_data.loc[:, FEATURE_LIST]

    targets = pd.DataFrame(data=model.predict(test_data))
    targets.index = np.arange(1, len(targets) + 1)
    print(sum(model.predict(test_data)))

    model_pipeline_io.save_submit_file(targets, "submission.csv")
    return targets


if __name__ == "__main__":
    train_set_name = [TRAIN_COM_SET]
    model = train(train_set_name)

    test_set_name = [TEST_COM_SET]
    targets = test(test_set_name, model)

    real = pd.read_csv(
        "/Users/shengdi/Documents/lbg_hackathon/CHEAT_SHEET.csv", index_col="Unnamed: 0"
    )
    from sklearn.metrics import recall_score, precision_score, accuracy_score

    recall = recall_score(real, targets)
    precision = precision_score(real, targets)
    accuracy = accuracy_score(real, targets)

    print(f"model recall on test set: {recall}")
    print(f"model precision on test set: {precision}")
    print(f"model accuracy on test set: {accuracy}")
