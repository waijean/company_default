import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np

from cmpny_bc_cls.utils.feature_imputation import fill_null_with_zeros


@pytest.mark.parametrize(
    "inputs,expected",
    [
        (
            pd.DataFrame(
                data=np.array([[1, 2, None], [None, 3, 4]], dtype=float),
                columns=["col1", "col2", "col3"],
            ),
            pd.DataFrame(
                data=np.array([[1, 2, 0], [0, 3, 4]], dtype=float),
                columns=["col1", "col2", "col3"],
            ),
        ),
    ],
)
def test_fill_null_with_zeros(inputs, expected):
    results = fill_null_with_zeros(inputs, ["col1", "col3"])
    assert_frame_equal(results, expected)
