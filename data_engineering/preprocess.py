import pandas as pd


def derive_total_assets_from_log_total_assets(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{"TOTAL_ASSETS": 10 ** df["logarithm of total assets"],})


def derive_features_from_total_assets(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        **{
            "WORKING_CAPITAL1": calculate_working_capital1(df),
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
            "EQUITY": calculate_equity(df),
            "RETAINED_EARNINGS": calculate_retained_earnings(df),
        }
    )


def derive_features_from_gross_profit(df):
    return df.assign(
        **{"SALES1": calculate_sales1(df), "INTEREST": calculate_interest(df),}
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
        }
    )


def derive_total_costs_from_total_sales(df):
    return df.assign(
        **{"TOTAL_COSTS": df["TOTAL_SALES"] * df["total costs /total sales"]}
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


def calculate_equity(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["equity / total assets"]


def calculate_retained_earnings(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["retained earnings / total assets"]


# def calculate_total_costs(df: pd.DataFrame) -> pd.Series:
#     return df["TOTAL_SALES"] * df["total costs /total sales"]


def calculate_sales1(df: pd.DataFrame) -> pd.Series:
    return df["GROSS_PROFIT"] / df["gross profit / sales"]


def calculate_interest(df: pd.DataFrame) -> pd.Series:
    return (
        df["TOTAL_ASSETS"] * df["(gross profit + interest) / total assets"]
        - df["GROSS_PROFIT"]
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


def calculate_book_value_of_equity(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_LIABILITIES"] * df["book value of equity / total liabilities"]


def calculate_depreciation1(df: pd.DataFrame) -> pd.Series:
    return (
        df["TOTAL_LIABILITIES"]
        * df["(gross profit + depreciation) / total liabilities"]
        - df["GROSS_PROFIT"]
    )


def calculate_depreciation2(df: pd.DataFrame) -> pd.Series:
    return (
        df["SALES"] * df["(gross profit + depreciation) / sales"] - df["GROSS_PROFIT"]
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


def calculate_cost_of_products_sold(df: pd.DataFrame) -> pd.Series:
    return -(df["SALES"] * df["(sales - cost of products sold) / sales"] - df["SALES"])


def calculate_inventory(df: pd.DataFrame) -> pd.Series:
    return df["SALES"] * df["(inventory * 365) / sales"] / 365


def calculate_receivables(df: pd.DataFrame) -> pd.Series:
    return df["SALES"] * df["(receivables * 365) / sales"] / 365


def calculate_cash(df: pd.DataFrame) -> pd.Series:
    return -(
        df["SALES"] * df["(total liabilities - cash) / sales"] - df["TOTAL_LIABILITIES"]
    )


def calculate_previous_year_sales(df: pd.DataFrame) -> pd.Series:
    return df["SALES"] / df["sales (n) / sales (n-1)"]


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


def calculate_working_capital1(df: pd.DataFrame) -> pd.Series:
    return df["TOTAL_ASSETS"] * df["working capital / total assets"]


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


def third_layer_derivation(df):
    return (
        df.pipe(derive_total_costs_from_total_sales)
        .pipe(derive_features_from_gross_profit)
        .pipe(derive_financial_expenses_from_profit_on_operating_expenses)
        .pipe(derive_features_from_total_liabilities)
        .pipe(derive_features_from_equity)
    )


def fourth_layer_derivation(df):
    return df.pipe(derive_features_from_sales1).pipe(
        derive_extraordinary_items_from_financial_expenses
    )


def assign_number_of_null_values_per_row(df):
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
#  1. refactor this into di's data pipeline
#  2. standardize naming for 'logarithm of total assets', 'bankruptcy_label' and 'working_capital2'
#  (i.e. from orignal feature).
#  3. rotation receivables + inventory turnover in days not added to the script -- do we need to include?
preprocessed_df = (
    df.pipe(derive_total_assets_from_log_total_assets)
    .pipe(derive_features_from_total_assets)
    .pipe(third_layer_derivation)
    .pipe(fourth_layer_derivation)
)
