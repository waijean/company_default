import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from utils import feature_imputation


@pytest.fixture
def input_df():
    df = pd.DataFrame(
        data=np.array(
            [[1, 2, None], [None, 3, 4], [10, 11, 20], [7, 10, 3]], dtype=float
        ),
        columns=["col1", "col2", "col3"],
    )
    return df


def test_fill_null_with_zeros(input_df):
    expected = pd.DataFrame(
        data=np.array([[1, 2, 0], [0, 3, 4], [10, 11, 20], [7, 10, 3]], dtype=float),
        columns=["col1", "col2", "col3"],
    )

    results = feature_imputation.fill_null_with_mean(input_df, ["col1", "col3"])
    assert_frame_equal(results, expected)


def test_fill_null_with_mean(input_df):
    expected = pd.DataFrame(
        data=np.array([[1, 2, 9], [6, 3, 4], [10, 11, 20], [7, 10, 3]], dtype=float),
        columns=["col1", "col2", "col3"],
    )

    results = feature_imputation.fill_null_with_mean(input_df, ["col1", "col3"])
    assert_frame_equal(results, expected)


def test_fill_null_with_midean(input_df):
    expected = pd.DataFrame(
        data=np.array([[1, 2, 4], [7, 3, 4], [10, 11, 20], [7, 10, 3]], dtype=float),
        columns=["col1", "col2", "col3"],
    )

    results = feature_imputation.fill_null_with_midean(input_df, ["col1", "col3"])
    assert_frame_equal(results, expected)
