def remove_null_and_duplicate_rows(df):
    return (
        df.pipe(remove_companies_with_half_missing_data)
        .pipe(remove_companies_with_null_assets)
        .pipe(remove_duplicate_rows)
    )


def calculate_number_of_null_values_per_row(df):
    return df.assign(**{"NULL_VALUE_COUNT": df.isna().sum(axis=1)})


def remove_companies_with_half_missing_data(df):
    # todo: maybe we should explicitly remove the row using that company_id? or use the assign above to help filter
    df = calculate_number_of_null_values_per_row(df)
    col_count = len(df.columns)
    return df.loc[(df["NULL_VALUE_COUNT"] < (col_count / 3))].drop(
        "NULL_VALUE_COUNT", axis=1
    )


def remove_companies_with_null_assets(df):
    if "TOTAL_ASSETS" in df.columns:
        # todo: maybe we should explicitly remove the row using that company_id? or use the assign above to help filter
        return df.loc[(~df["TOTAL_ASSETS"].isna())]
    else:
        return df


def remove_duplicate_rows(df):
    return df.drop_duplicates()


def remove_duplicate_columns(df):
    # todo: need to try different combination of cols to find other possible duplicates
    cols = list(df.columns)
    cols.remove("COMPANY_ID")
    return df.drop_duplicates(cols)
