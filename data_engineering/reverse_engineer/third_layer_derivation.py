import pandas as pd


def derive_third_layer(df):
    return (
        df.pipe(derive_total_costs_from_total_sales)
        .pipe(derive_features_from_gross_profit)
        .pipe(combine_sales)
        .pipe(derive_financial_expenses_from_profit_on_operating_expenses)
        .pipe(derive_features_from_total_liabilities)
        .pipe(derive_features_from_equity)
    )


def derive_total_costs_from_total_sales(df):
    return df.assign(
        **{"TOTAL_COSTS": df["TOTAL_SALES"] * df["total costs /total sales"]}
    )


def derive_features_from_gross_profit(df):
    return df.assign(**{"SALES1": calculate_sales1(df)})


def combine_sales(df):
    return df.assign(**{"SALES": df["SALES1"].combine_first(df["SALES2"])}).drop(
        ["SALES1", "SALES2"], axis=1
    )


def derive_financial_expenses_from_profit_on_operating_expenses(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.assign(
        **{
            "FINANCIAL_EXPENSES": df["PROFIT_ON_OPERATING_ACTIVITIES"]
            / df["profit on operating activities / financial expenses"]
        }
    )


def derive_features_from_total_liabilities(df):
    return df.assign(
        **{
            "BOOK_VALUE_OF_EQUITY": calculate_book_value_of_equity(df),
            "DEPRECIATION1": calculate_depreciation1(df),
            "CURRENT_ASSETS": calculate_current_assets(df),
            "OPERATING_EXPENSES": calculate_operating_expenses(df),
        }
    )


def derive_features_from_equity(df):
    return df.assign(
        **{
            "FIXED_ASSETS": calculate_fixed_assets(df),
            "SHARE_CAPITAL": calculate_share_capital(df),
            "LONG_TERM_LIABILITIES": calculate_long_term_liabilities(df),
        }
    )


def calculate_sales1(df: pd.DataFrame) -> pd.Series:
    return df["GROSS_PROFIT"] / df["gross profit / sales"]


def calculate_book_value_of_equity(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_LIABILITIES"] * df["book value of equity / total liabilities"]


def calculate_depreciation1(df: pd.DataFrame) -> pd.Series:
    return (
        df["TOTAL_LIABILITIES"]
        * df["(gross profit + depreciation) / total liabilities"]
        - df["GROSS_PROFIT"]
    )


def calculate_current_assets(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_LIABILITIES"] * df["current assets / total liabilities"]


def calculate_operating_expenses(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_LIABILITIES"] * df["operating expenses / total liabilities"]


def calculate_fixed_assets(df: pd.DataFrame) -> pd.Series:
    return df["EQUITY"] / df["equity / fixed assets"]


def calculate_share_capital(df: pd.DataFrame) -> pd.Series:
    return -(
        df["TOTAL_ASSETS"] * df["(equity - share capital) / total assets"]
        - df["EQUITY"]
    )


def calculate_long_term_liabilities(df: pd.DataFrame) -> pd.Series:
    return df["EQUITY"] * df["long-term liabilities / equity"]
