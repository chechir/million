import numpy as np
import pandas as pd
from million.features import quasi_medians, grouped_quasi_medians


def test_quasi_medians():
    train_ixs = np.array([1, 1, 1, 0, 0, 0]).astype(bool)
    v = np.array([0, 1, 4, -1, 2, 5])
    expected = np.array([2.5, 2.0, 0.5, 1, 1, 1])
    group_df = pd.DataFrame({
        'logerror': v,
        'train_ixs': train_ixs
        })
    result = quasi_medians(group_df)
    assert np.allclose(expected, result)


def test_grouped_quasi_medians():
    train_ixs = np.array([1, 1, 1, 1, 1, 0]).astype(bool)
    df = pd.DataFrame({
        'regionidneighborhood': [1, 1, 1, 2, 2, 2],
        'logerror': [0, 1, 4, -1, 2, 5]
        })

    expected = np.array([2.5, 2.0, 0.5, 2, -1, 0.5])
    result = grouped_quasi_medians(df, train_ixs, groupby='regionidneighborhood')
    assert np.allclose(expected, result)
