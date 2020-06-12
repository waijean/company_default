import pandas as pd


def derive_fourth_layer(df):
    return (
        df.pipe(derive_features_from_sales1)
        .pipe(combine_depreciation)
        .pipe(derive_extraordinary_items_from_financial_expenses)
    )


def derive_features_from_sales1(df):
    return df.assign(
        **{
            "COST_OF_PRODUCTS_SOLD": calculate_cost_of_products_sold(df),
            "INVENTORY": calculate_inventory(df),
            "RECEIVABLES": calculate_receivables(df),
            "DEPRECIATION2": calculate_depreciation2(df),
            "CASH": calculate_cash(df),
            "PREVIOUS_YEAR_SALES": calculate_previous_year_sales(df),
        }
    )


def combine_depreciation(df):
    return df.assign(
        **{"DEPRECIATION": df["DEPRECIATION1"].combine_first(df["DEPRECIATION2"])}
    ).drop(["DEPRECIATION1", "DEPRECIATION2"], axis=1)


def derive_extraordinary_items_from_financial_expenses(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return df.assign(
        **{
            "EXTRAORDINARY_ITEMS": df["TOTAL_ASSETS"]
            * df[
                "(gross profit + extraordinary items + financial expenses) / total assets"
            ]
            - df["GROSS_PROFIT"]
            - df["FINANCIAL_EXPENSES"]
        }
    )


def calculate_cost_of_products_sold(df: pd.DataFrame) -> pd.Series:
    return -(df["SALES"] * df["(sales - cost of products sold) / sales"] - df["SALES"])


def calculate_inventory(df: pd.DataFrame) -> pd.Series:
    return df["SALES"] * df["(inventory * 365) / sales"] / 365


def calculate_receivables(df: pd.DataFrame) -> pd.Series:
    return df["SALES"] * df["(receivables * 365) / sales"] / 365


def calculate_depreciation2(df: pd.DataFrame) -> pd.Series:
    return (
        df["SALES"] * df["(gross profit + depreciation) / sales"] - df["GROSS_PROFIT"]
    )


def calculate_cash(df: pd.DataFrame) -> pd.Series:
    return -(
        df["SALES"] * df["(total liabilities - cash) / sales"] - df["TOTAL_LIABILITIES"]
    )


def calculate_previous_year_sales(df: pd.DataFrame) -> pd.Series:
    return df["SALES"] / df["sales (n) / sales (n-1)"]
