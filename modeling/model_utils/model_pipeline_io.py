import pandas as pd
import yaml
import os

from model_utils import drop_duplicate


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


def get_training_set(train_set_name: list):
    train_data = pd.concat(
        [read_train_file(name).iloc[:, :-1] for name in train_set_name], axis=1,
    )
    train_data_with_target = pd.concat(
        [train_data, read_train_file(train_set_name[0]).iloc[:, -1]], axis=1
    )

    train_data_with_target = drop_duplicate.remove_null_and_duplicate_rows(
        train_data_with_target
    )
    return train_data_with_target.iloc[:, :-1], train_data_with_target.iloc[:, -1]
