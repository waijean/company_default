import pandas as pd

from utils import pipeline_io
from utils.preprocessing import (
    remove_companies_with_many_null_values,
    remove_duplicates,
    get_raw_values,
    fill_null_with_zeros,
    clip_extreme_value,
)


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the labelled training set.
    """
    COLS_IMPUTE_ZEROS = list(df.columns)
    COLS_TO_CLIP = list(df.columns)
    return (
        df.pipe(remove_duplicates)
        .pipe(remove_companies_with_many_null_values, thres=55)
        .pipe(fill_null_with_zeros, columns=COLS_IMPUTE_ZEROS)
        .pipe(clip_extreme_value, columns=COLS_TO_CLIP)
    )


def clean_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the unlabelled test set.

    Basically the same steps as clean_train() but without dropping any rows since we need to give a prediction for
    all the companies in the test set
    """
    COLS_IMPUTE_ZEROS = list(df.columns)
    COLS_TO_CLIP = list(df.columns)
    return df.pipe(fill_null_with_zeros, columns=COLS_IMPUTE_ZEROS).pipe(
        clip_extreme_value, columns=COLS_TO_CLIP
    )


def create_combined_df(ratio_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the raw and ratio df

    Need to take care of the duplicate target columns in both tables
    """
    return pd.concat(
        [raw_df.iloc[:, :-1], ratio_df.iloc[:, :-1], raw_df.iloc[:, -1]], axis=1
    )


def run(type: str):
    """
    Preprocessing pipeline consists of following steps:
        1. Clean the ratio df
        2. Get the raw values from ratio df
        3. Concat both ratio and raw df

    Args:
        type: A flag {'train', 'test} to decide whether to pre-process the train file or the test file.

    Returns: Write three separates files (raw, ratio and combined) to the output directory.

    """
    ratio_df = pipeline_io.read_file(type)
    if type == "test":
        cleaned_ratio_df = clean_test(ratio_df)
    elif type == "train":
        cleaned_ratio_df = clean_train(ratio_df)
    else:
        raise TypeError("Type has to be either 'test' or 'train'")

    # create raw_df
    raw_df = get_raw_values(cleaned_ratio_df)

    # create combined_df
    combined_df = create_combined_df(cleaned_ratio_df, raw_df)
    combined_df = clean_test(combined_df)

    pipeline_io.create_output_dir()
    pipeline_io.save_file(cleaned_ratio_df, f"ratio_{type}")
    pipeline_io.save_file(raw_df, f"raw_{type}")
    pipeline_io.save_file(combined_df, f"combined_{type}")


if __name__ == "__main__":
    for type in ["train", "test"]:
        run(type=type)
