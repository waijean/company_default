import pandas as pd

from cmpny_bc_cls.utils import pipeline_io
from cmpny_bc_cls.utils import feature_imputation

COLS_IMPUTE_ZEROS = ["gross profit (in 3 years) / total assets"]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = feature_imputation.fill_null_with_zeros(df, COLS_IMPUTE_ZEROS)
    return df


def run():
    train_df = pipeline_io.read_train_file()
    print(train_df.head())

    transformed_df = transform(train_df)

    pipeline_io.save_clean_train_file(transformed_df)


if __name__ == "__main__":
    run()
