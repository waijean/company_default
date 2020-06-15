import pandas as pd

from utils import feature_imputation, pipeline_io, preprocessing

from utils.other_derivation import (
    derive_total_assets_from_log_total_assets,
    derive_short_term_securities_from_cash,
    rename_column,
)
from utils.fourth_layer_derivation import derive_fourth_layer
from utils.second_layer_derivation import derive_second_layer
from utils.third_layer_derivation import derive_third_layer

COLS_IMPUTE_ZEROS = ["gross profit (in 3 years) / total assets"]
COLS_SKEWED = ["TOTAL_ASSETS"]


def get_raw_values(ratio_df):
    """
    - First and fifth layers are only deriving one column
    - Exclude rotation receivables + inventory turnover in days since they are derived from receivable days and
    inventory days which we can calculate from raw values

    Args:
        ratio_df:

    Returns: Dataframe with 32 columns of raw values

    """
    raw_values_df = (
        ratio_df.pipe(derive_total_assets_from_log_total_assets)
        .pipe(derive_second_layer)
        .pipe(derive_third_layer)
        .pipe(derive_fourth_layer)
        .pipe(derive_short_term_securities_from_cash)
        .pipe(rename_column)
    )

    return raw_values_df.drop(ratio_df.columns, axis=1)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    COLS_SKEWED = list(df.columns[:-1])
    df = preprocessing.fix_skewed_data(df, COLS_SKEWED)
    COLS_IMPUTE_ZEROS = list(df.columns[:-1])
    df = feature_imputation.fill_null_with_zeros(df, COLS_IMPUTE_ZEROS)
    return df


def run(clean_ratio=True, get_raw=True):
    """
    Run two different jobs
    1. Clean the original ratio df
    2. Get the raw values from original ratio df
    """
    pipeline_io.create_output_dir()
    train_df = pipeline_io.read_train_file()
    test_df = pipeline_io.read_test_file()

    if clean_ratio:
        transformed_df = transform(train_df)
        transformed_df_test = transform(test_df)

        pipeline_io.save_raw_values_file(transformed_df, "cleaned_ratio_train.csv")
        pipeline_io.save_raw_values_file(transformed_df_test, "cleaned_ratio_test.csv")

    if get_raw:
        raw_values_df = get_raw_values(train_df)
        transformed_df = transform(raw_values_df)

        raw_values_df_test = get_raw_values(test_df)
        transformed_df_test = transform(raw_values_df_test)

        # the raw values df should have 32 features columns and 1 target column
        assert raw_values_df.shape[1] == 32 + 1
        assert raw_values_df_test.shape[1] == 32 + 1

        pipeline_io.save_raw_values_file(transformed_df, "cleaned_raw_train.csv")
        pipeline_io.save_raw_values_file(transformed_df_test, "cleaned_raw_test.csv")


if __name__ == "__main__":
    run(clean_ratio=False)
