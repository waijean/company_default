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
