import pandas as pd


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
