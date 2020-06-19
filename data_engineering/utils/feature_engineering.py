import pandas as pd

from utils.fourth_layer_derivation import derive_fourth_layer
from utils.other_derivation import (
    derive_total_assets_from_log_total_assets,
    derive_short_term_securities_from_cash,
    rename_column,
)
from utils.second_layer_derivation import derive_second_layer
from utils.third_layer_derivation import derive_third_layer


def fill_null_with_zeros(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_new = df.copy()
    df_new[columns] = df_new[columns].fillna(0)
    return df_new


def fill_null_with_mean(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_new = df.copy()
    df_new[columns] = df_new[columns].fillna(df[columns].mean())
    return df_new


def fill_null_with_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df_new = df.copy()
    df_new[columns] = df_new[columns].fillna(df[columns].median())
    return df_new


def remove_companies_with_many_null_values(
    df: pd.DataFrame, thres: int
) -> pd.DataFrame:
    df = df.assign(**{"NULL_VALUE_COUNT": df.isna().sum(axis=1)})
    return df.loc[(df["NULL_VALUE_COUNT"] < thres)].drop("NULL_VALUE_COUNT", axis=1)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # todo: need to try different combination of cols to find other possible duplicates
    cols = list(df.columns)
    return df.drop_duplicates(cols)


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
