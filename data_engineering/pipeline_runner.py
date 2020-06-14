import pandas as pd

from utils import feature_imputation, pipeline_io

from utils.other_derivation import (
    derive_total_assets_from_log_total_assets,
    derive_short_term_securities_from_cash,
    rename_column,
)
from utils.fourth_layer_derivation import derive_fourth_layer
from utils.second_layer_derivation import derive_second_layer
from utils.third_layer_derivation import derive_third_layer

TRAIN = "train"
TEST = "test"
COLS_IMPUTE_ZEROS = ["gross profit (in 3 years) / total assets"]


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
    df = feature_imputation.fill_null_with_zeros(df, COLS_IMPUTE_ZEROS)
    return df


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

    transformed_df = transform(df)

    raw_values_df = get_raw_values(df)

    pipeline_io.save_file(transformed_df, output_files[f"ratio_{TRAIN}"])

    # the raw values df should have 32 features columns and 1 target column
    assert raw_values_df.shape[1] == 32 + 1
    pipeline_io.save_file(raw_values_df, output_files[f"raw_{TRAIN}"])


if __name__ == "__main__":
    run()
