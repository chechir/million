import numpy as np
import pandas as pd
from million.features import quasi_medians, grouped_quasi_medians

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


def quasi_medians_new(x):
    v, train_ixs = x['logerror'], x['train_ixs'].astype(bool)
    v_train = v[train_ixs]
    train_length = len(v_train)
    result = np.nan * np.zeros(len(v))
    result[~train_ixs] = np.median(v_train)
    i = 0
    for i in range(len(v_train)):
        other_ixs = np.ones(train_length, dtype=bool)
        other_ixs[i] = False
        result_train = result[train_ixs]
        result_train[~other_ixs] = np.median(v_train[other_ixs])
        result[train_ixs] = result_train
    print 'processed {} elements'.format(i)
    return result


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

def test_speedup_quasi_medians(df):
    from million import data
    df = data.load_data(from_cache=True)
    train_targets = df['logerror'].values
    train_ixs, test_ixs = data.get_ixs(train_targets)

    df = features.add_features(df, train_ixs)

