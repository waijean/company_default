import pandas as pd


def derive_second_layer(df: pd.DataFrame) -> pd.DataFrame:
    return df.pipe(derive_features_from_total_assets).pipe(combine_working_capital)


def derive_features_from_total_assets(df):
    return df.assign(
        **{
            "TOTAL_SALES": calculate_total_sales(df),
            "SALES2": calculate_sales2(df),
            "GROSS_PROFIT": calculate_gross_profit(df),
            "NET_PROFIT": calculate_net_profit(df),
            "EBIT": calculate_ebit(df),
            "EBITDA": calculate_ebitda(df),
            "GROSS_PROFIT_IN_3_YEARS": calculate_gross_profit_in_3_years(df),
            "PROFIT_ON_OPERATING_ACTIVITIES": calculate_profit_on_operating_activities(
                df
            ),
            "PROFIT_ON_SALES": calculate_profit_on_sales(df),
            "CONSTANT_CAPITAL": calculate_constant_capital(df),
            "TOTAL_LIABILITIES": calculate_total_liabilities(df),
            "SHORT_TERM_LIABILITIES": calculate_short_term_liabilities(df),
            "WORKING_CAPITAL1": calculate_working_capital1(df),
            "EQUITY": calculate_equity(df),
            "RETAINED_EARNINGS": calculate_retained_earnings(df),
        }
    )


def combine_working_capital(df):
    return df.assign(
        **{
            "WORKING_CAPITAL": df["WORKING_CAPITAL1"].combine_first(
                df["working capital"]
            )
        }
    ).drop(["WORKING_CAPITAL1"], axis=1)


def calculate_total_sales(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["total sales / total assets"]


def calculate_sales2(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["sales / total assets"]


def calculate_gross_profit(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["gross profit / total assets"]


def calculate_net_profit(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["net profit / total assets"]


def calculate_ebit(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["EBIT / total assets"]


def calculate_ebitda(df: pd.DataFrame) -> pd.Series:
    return (
        df["TOTAL_ASSETS"]
        * df["EBITDA (profit on operating activities - depreciation) / total assets"]
    )


def calculate_gross_profit_in_3_years(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["gross profit (in 3 years) / total assets"]


def calculate_profit_on_operating_activities(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["profit on operating activities / total assets"]


def calculate_profit_on_sales(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["profit on sales / total assets"]


def calculate_constant_capital(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["constant capital / total assets"]


def calculate_total_liabilities(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["total liabilities / total assets"]


def calculate_short_term_liabilities(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["short-term liabilities / total assets"]


def calculate_working_capital1(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["working capital / total assets"]


def calculate_equity(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["equity / total assets"]


def calculate_retained_earnings(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["retained earnings / total assets"]
