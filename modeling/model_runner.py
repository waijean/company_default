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
TRAIN_RATIO_SET = "ratio_train.csv"  # "cleaned_ratio_train.csv"
TRAIN_RAW_SET = "raw_train.csv"  # "cleaned_raw_train.csv"
TEST_RATIO_SET = "ratio_test.csv"  # "cleaned_ratio_test.csv"
TEST_RAW_SET = "raw_test.csv"  # "cleaned_raw_test.csv"
TRAIN_COM_SET = "combined_train.csv"
TEST_COM_SET = "combined_test.csv"


def train(train_set_name: list):
    train_data, target = model_pipeline_io.get_training_set(train_set_name)
    print(train_data.index[np.isinf(train_data).any(1)])
    pipe = make_pipeline(
        StandardScaler(),
        RandomUnderSampler(random_state=RANDOM_STATE),
        BalancedRandomForestClassifier(**hyperparameter_const.COMBINE_DATA),
    )

    pipe.fit(train_data, target)
    return pipe


def test(test_set_name: list, model: RandomForestClassifier):
    test_data = model_pipeline_io.get_test_set(test_set_name)

    targets = pd.DataFrame(data=model.predict(test_data))
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
