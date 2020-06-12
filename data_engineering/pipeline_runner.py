import pandas as pd

from utils import feature_imputation, pipeline_io

COLS_IMPUTE_ZEROS = ["gross profit (in 3 years) / total assets"]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    COLS_IMPUTE_ZEROS = list(df.columns)
    df = feature_imputation.fill_null_with_zeros(df, COLS_IMPUTE_ZEROS)
    return df


def run():
    train_df = pipeline_io.read_train_file()
    print(train_df.head())

    transformed_df = transform(train_df)

    pipeline_io.save_clean_train_file(transformed_df)


if __name__ == "__main__":
    run()
