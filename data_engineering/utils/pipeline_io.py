import pandas as pd
import yaml
import os
import git

repo = git.Repo(".", search_parent_directories=True)
ROOT_DIR_PATH = repo.working_tree_dir


def get_para() -> dict:
    file_path = os.path.join(ROOT_DIR_PATH, "conf/parameters.yaml")
    print(f"path to the config file: {file_path}")
    with open(file_path) as file:
        paras = yaml.load(file, Loader=yaml.FullLoader)
    return paras


def read_train_file() -> pd.DataFrame:
    paras = get_para()
    file_path = os.path.join(paras["path_to_input_dir"], "train.csv")
    df = pd.read_csv(file_path, index_col="company_id")
    return df


def read_test_file() -> pd.DataFrame:
    paras = get_para()
    file_path = os.path.join(paras["path_to_input_dir"], "test.csv")
    df = pd.read_csv(file_path, index_col="company_id")
    return df


def save_clean_train_file(df: pd.DataFrame):
    paras = get_para()
    file_path = os.path.join(paras["path_to_output_dir"], "cleaned_ratio_train.csv")
    print(f"save processed data set to: {file_path}")
    df.to_csv(file_path)


def save_raw_values_file(df: pd.DataFrame):
    paras = get_para()
    file_path = os.path.join(paras["path_to_output_dir"], "cleaned_raw_train.csv")
    print(f"save processed data set to: {file_path}")
    df.to_csv(file_path)


def create_output_dir():
    paras = get_para()
    file_path = paras["path_to_output_dir"]
    if not os.path.exists(file_path):
        print(f"create output directory: {file_path}")
        os.makedirs(file_path)
    else:
        print(f"skipped creating output directory")
