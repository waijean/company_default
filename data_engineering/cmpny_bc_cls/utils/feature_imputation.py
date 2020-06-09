import pandas as pd


def fill_null_with_zeros(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df[columns] = df[columns].fillna(0)
    return df
