import pandas as pd


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
