import pandas as pd
import yaml
import os

from modeling.utils.processing import remove_null_and_duplicate_rows


def get_para() -> dict:
    file_path = os.path.join(os.path.abspath(""), "..", "conf/parameters.yaml")
    print(f"path to the config file: {file_path}")
    with open(file_path) as file:
        paras = yaml.load(file, Loader=yaml.FullLoader)
    return paras


def read_train_file(filename: str) -> pd.DataFrame:
    paras = get_para()
    filepath = os.path.join(paras["path_to_output_dir"], filename)
    df = pd.read_csv(filepath, index_col="company_id")
    return df


def read_clean_train_file():
    paras = get_para()
    file_path = os.path.join(paras["path_to_output_dir"], "cleaned_ratio_train.csv")
    print(f"Read cleaned ratio data set from: {file_path}")
    df = pd.read_csv(file_path, index_col="company_id")
    return df


def read_raw_values_file():
    paras = get_para()
    file_path = os.path.join(paras["path_to_output_dir"], "cleaned_raw_train.csv")
    print(f"Read cleaned raw data set from: {file_path}")
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
    paras = get_para()
    file_path = os.path.join(paras["path_to_output_dir"], file_name)
    print(f"save submit targets to: {file_path}")
    df.to_csv(file_path)
