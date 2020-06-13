import pandas as pd
import yaml
import os


def get_para() -> dict:
    file_path = os.path.join(os.path.abspath(""), "..", "conf/parameters.yaml")
    print(f"path to the config file: {file_path}")
    with open(file_path) as file:
        paras = yaml.load(file, Loader=yaml.FullLoader)
    return paras


def read_train_file() -> pd.DataFrame:
    paras = get_para()
    df = pd.read_csv(paras["path_to_clean_train_file"], index_col="company_id")
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
