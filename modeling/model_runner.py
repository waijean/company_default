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

from model_utils import model_pipeline_io, hyperparameter_const

RANDOM_STATE = 42
TRAIN_RATIO_SET = "cleaned_ratio_train.csv"
TRAIN_RAW_SET = "cleaned_raw_train.csv"
TEST_RATIO_SET = "cleaned_ratio_test.csv"
TEST_RAW_SET = "cleaned_raw_test.csv"


def train(train_set_name: list):
    train_data, target = model_pipeline_io.get_training_set(train_set_name)

    pipe = make_pipeline(
        StandardScaler(),
        RandomUnderSampler(random_state=RANDOM_STATE),
        BalancedRandomForestClassifier(**hyperparameter_const.COMBINE_DATA),
    )

    pipe.fit(train_data, target)
    return pipe[-1]


def test(test_set_name: list, model: RandomForestClassifier):
    test_data = model_pipeline_io.get_test_set(test_set_name)

    targets = pd.DataFrame(data=model.predict(test_data))
    print(sum(model.predict(test_data)))

    model_pipeline_io.save_submit_file(targets, "submission.csv")


if __name__ == "__main__":
    train_set_name = [TRAIN_RATIO_SET, TRAIN_RAW_SET]
    model = train(train_set_name)

    test_set_name = [TEST_RATIO_SET, TEST_RAW_SET]
    test(test_set_name, model)
