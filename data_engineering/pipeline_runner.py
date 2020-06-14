import pandas as pd

from utils import pipeline_io
from utils.feature_engineering import (
    remove_companies_with_many_null_values,
    remove_duplicates,
    fill_null_with_zeros,
    get_raw_values,
)

TRAIN = "train"
TEST = "test"
COLS_IMPUTE_ZEROS = ["gross profit (in 3 years) / total assets"]


def transform_ratio(df: pd.DataFrame) -> pd.DataFrame:
    # todo: to be expanded
    return (
        df.pipe(fill_null_with_zeros, columns=COLS_IMPUTE_ZEROS)
        .pipe(remove_duplicates)
        .pipe(remove_companies_with_many_null_values, thres=55)
    )


def transform_raw(df: pd.DataFrame) -> pd.DataFrame:
    # todo: to be expanded
    return (
        df.pipe(get_raw_values)
        .pipe(remove_duplicates)
        .pipe(remove_companies_with_many_null_values, thres=25)
    )


def run():
    """
    Run two different jobs
    1. Clean the original ratio df
    2. Get the raw values from original ratio df
    """
    output_files = {
        "ratio_train": "ratio_train.csv",
        "ratio_test": "ratio_test.csv",
        "raw_train": "raw_train.csv",
        "raw_test": "raw_test.csv",
    }

    pipeline_io.create_output_dir()

    df = pipeline_io.read_file(TRAIN)
    transformed_ratio_df = transform_ratio(df)
    transformed_raw_df = transform_raw(df)
    pipeline_io.save_file(transformed_ratio_df, output_files[f"ratio_{TRAIN}"])

    # the raw values df should have 32 features columns and 1 target column
    assert transformed_raw_df.shape[1] == 32 + 1
    pipeline_io.save_file(transformed_raw_df, output_files[f"raw_{TRAIN}"])


if __name__ == "__main__":
    run()
