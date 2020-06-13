import pandas as pd
import yaml
import os


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
