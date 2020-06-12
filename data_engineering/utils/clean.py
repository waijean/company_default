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
