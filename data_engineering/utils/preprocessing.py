import pandas as pd
import numpy as np

from reverse_engineer.fourth_layer_derivation import derive_fourth_layer
from reverse_engineer.other_derivation import (
    derive_total_assets_from_log_total_assets,
    derive_short_term_securities_from_cash,
    rename_column,
)
from reverse_engineer.second_layer_derivation import derive_second_layer
from reverse_engineer.third_layer_derivation import derive_third_layer


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


def get_raw_values(ratio_df: pd.DataFrame) -> pd.DataFrame:
    """
    - First and fifth layers are only deriving one column
    - Exclude rotation receivables + inventory turnover in days since they are derived from receivable days and
    inventory days which we can calculate from raw values

    Args:
        ratio_df:

    Returns: Dataframe with 32 columns of raw values

    """
    raw_df = (
        ratio_df.pipe(derive_total_assets_from_log_total_assets)
        .pipe(derive_second_layer)
        .pipe(derive_third_layer)
        .pipe(derive_fourth_layer)
        .pipe(derive_short_term_securities_from_cash)
        .pipe(rename_column)
    )

    raw_df = raw_df.drop(ratio_df.columns, axis=1)

    # the raw values df should have 32 features columns and 1 target column
    assert raw_df.shape[1] == 32 + 1

    return raw_df


def clip_extreme_value(df: pd.DataFrame, columns: list):
    for col in columns:
        mask = ~np.isinf(df[col])
        mask_inf = df[col] == np.inf
        mask_inf_neg = df[col] == -np.inf
        df.loc[mask_inf, col] = df.loc[mask, col].max()
        df.loc[mask_inf_neg, col] = df.loc[mask, col].min()
    return df


def take_log(x):
    if x > 0.0000000001:
        return np.log10(x)
    else:
        return -10


def fix_skewed_data(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].apply(take_log)
    return df
