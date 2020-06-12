import pandas as pd

from data_engineering.utils.fourth_layer_derivation import derive_fourth_layer
from data_engineering.utils.second_layer_derivation import derive_second_layer
from data_engineering.utils.third_layer_derivation import derive_third_layer


def derive_total_assets_from_log_total_assets(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{"TOTAL_ASSETS": 10 ** df["logarithm of total assets"],})


def derive_short_term_securities_from_cash(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        **{
            "SHORT_TERM_SECURITIES": (
                df[
                    "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365"
                ]
                / 365
            )
            * (df["OPERATING_EXPENSES"] - df["DEPRECIATION"])
            - df["CASH"]
            - df["RECEIVABLES"]
            + df["SHORT_TERM_LIABILITIES"]
        }
    )


def rename_column(df):
    return df.assign(**{"BANKRUPTCY_LABEL": df["bankruptcy_label"],})


def calculate_number_of_null_values_per_row(df):
    return df.assign(**{"NULL_VALUE_COUNT": df.isna().sum(axis=1)})


def remove_companies_with_null_values(df):
    # todo: maybe we should explicitly remove the row using that company_id? or use the assign above to help filter
    return df.loc[(~df["TOTAL_ASSETS"].isna())]


def remove_duplicates(df):
    # todo: need to try different combination of cols to find other possible duplicates
    cols = list(df.columns)
    cols.remove("COMPANY_ID")
    return df.drop_duplicates(cols)


# todo:
#  1. standardize naming for 'logarithm of total assets', 'bankruptcy_label' and 'working_capital2'
#  (i.e. from orignal feature).


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
