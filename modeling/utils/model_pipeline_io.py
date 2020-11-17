import git
import pandas as pd
import os

from modeling.utils.processing import remove_null_and_duplicate_rows

repo = git.Repo(".", search_parent_directories=True)
ROOT_DIR_PATH = repo.working_tree_dir


def read_train_file(filename: str) -> pd.DataFrame:
    filepath = os.path.join(ROOT_DIR_PATH, "data/output", filename)
    df = pd.read_csv(filepath, index_col="company_id")
    return df


def read_clean_train_file():
    file_path = os.path.join(ROOT_DIR_PATH, "data/output", "cleaned_ratio_train.csv")
    print(f"Reading cleaned ratio data set from: {file_path}")
    df = pd.read_csv(file_path, index_col="company_id")
    return df


def read_raw_values_file():
    file_path = os.path.join(ROOT_DIR_PATH, "data/output" "cleaned_raw_train.csv")
    print(f"Reading cleaned raw data set from: {file_path}")
    df = pd.read_csv(file_path, index_col="company_id")
    return df


def get_training_set(train_set_name: list):
    train_data = pd.concat(
        [read_train_file(name).iloc[:, :-1] for name in train_set_name], axis=1,
    )
    train_data_with_target = pd.concat(
        [train_data, read_train_file(train_set_name[0]).iloc[:, -1]], axis=1
    )

    train_data_with_target = remove_null_and_duplicate_rows(train_data_with_target)
    return train_data_with_target.iloc[:, :-1], train_data_with_target.iloc[:, -1]


def get_test_set(test_set_name: list):
    test_data = pd.concat(
        [read_train_file(name).iloc[:, :-1] for name in test_set_name], axis=1,
    )
    return test_data


def save_submit_file(df: pd.DataFrame, file_name: str):
    file_path = os.path.join(ROOT_DIR_PATH, "data/output", file_name)
    print(f"save submit targets to: {file_path}")
    df.to_csv(file_path)
