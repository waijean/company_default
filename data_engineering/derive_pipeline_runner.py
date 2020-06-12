import pandas as pd

from utils import pipeline_io

from data_engineering.derive.derive import get_raw_values


def run():
    train_df = pipeline_io.read_train_file()
    print(train_df.head())

    raw_values_df = get_raw_values(train_df)

    assert raw_values_df.shape[1] == 32 + 1

    pipeline_io.save_raw_values_file(raw_values_df)


if __name__ == "__main__":
    run()
