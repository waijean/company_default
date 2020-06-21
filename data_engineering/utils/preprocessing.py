import pandas as pd
import numpy as np


def take_log(x):
    if x > 0.0000000001:
        return np.log10(x)
    else:
        return -10


def fix_skewed_data(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].apply(take_log)
    return df
