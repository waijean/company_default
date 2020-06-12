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
    df = pd.read_csv(paras["path_to_train_file"], index_col="company_id")
    return df


def save_clean_train_file(df: pd.DataFrame):
    paras = get_para()
    file_path = paras["path_to_clean_train_file"]
    print(f"save processed data set to: {file_path}")
    df.to_csv(file_path)


def save_raw_values_file(df: pd.DataFrame):
    paras = get_para()
    dir_path = paras["path_to_output_dir"]
    file_path = os.path.join(dir_path, "raw_values.csv")
    print(f"save processed data set to: {file_path}")
    df.to_csv(file_path)
