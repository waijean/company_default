import pandas as pd
import os
import git

repo = git.Repo(".", search_parent_directories=True)
ROOT_DIR_PATH = repo.working_tree_dir


def read_file(filename: str) -> pd.DataFrame:
    file_path = os.path.join(ROOT_DIR_PATH, "data/input", f"{filename}.csv")
    print(f"read {filename} from: {file_path}")
    df = pd.read_csv(file_path, index_col="company_id")
    return df


def save_file(df: pd.DataFrame, filename: str):
    file_path = os.path.join(ROOT_DIR_PATH, "data/output", f"{filename}.csv")
    print(f"save {filename} to: {file_path}")
    df.to_csv(file_path)


def create_output_dir():
    file_path = ROOT_DIR_PATH
    if not os.path.exists(file_path):
        print(f"create output directory: {file_path}")
        os.makedirs(file_path)
    else:
        print(f"skipped creating output directory: {file_path}")
