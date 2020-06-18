import pandas as pd

from utils import pipeline_io
from utils.feature_engineering import (
    remove_companies_with_many_null_values,
    remove_duplicates,
    get_raw_values,
    fill_null_with_zeros,
)

TRAIN = "train"
TEST = "test"
COLS_IMPUTE_ZEROS = ["gross profit (in 3 years) / total assets"]


def transform_train(df: pd.DataFrame) -> pd.DataFrame:
    # todo: to be expanded
    return (
        df.pipe(remove_duplicates)
        .pipe(remove_companies_with_many_null_values, thres=55)
        .pipe(fill_null_with_zeros, columns=COLS_IMPUTE_ZEROS)
    )


def transform_test(df: pd.DataFrame) -> pd.DataFrame:
    # todo: to be expanded
    return df.pipe(fill_null_with_zeros, columns=COLS_IMPUTE_ZEROS)


def create_combined_df(ratio_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [raw_df.iloc[:, :-1], ratio_df.iloc[:, :-1], raw_df.iloc[:, -1]], axis=1
    )


def run(file: str):
    """
    Data engineering pipeline consisting of following steps:
    1. Clean the original ratio df
    2. Get the raw values from ratio df
    3. Concat both ratio and raw df
    :param file: str value, it's either train or test
    :return: processed ratio file, raw file and combined file
    """

    pipeline_io.create_output_dir()
    df = pipeline_io.read_file(file)
    if file == "test":
        transformed_ratio_df = transform_test(df)
    else:
        transformed_ratio_df = transform_train(df)
    transformed_raw_df = transformed_ratio_df.pipe(get_raw_values)
    pipeline_io.save_file(transformed_ratio_df, f"ratio_{file}")

    # the raw values df should have 32 features columns and 1 target column
    assert transformed_raw_df.shape[1] == 32 + 1
    pipeline_io.save_file(transformed_raw_df, f"raw_{file}")

    combined_df = create_combined_df(transformed_ratio_df, transformed_raw_df)
    pipeline_io.save_file(combined_df, f"combined_{file}")


if __name__ == "__main__":
    run("train")
