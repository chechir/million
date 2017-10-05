from functools import partial
import numpy as np
import pandas as pd
import seamless as ss
from million import tools as nptools

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


def add_features(df, train_ixs):
    qmedian_cols = ['regionidzip', 'propertylandusetypeid']
#    0.0656769900765 (BM)
    for col in qmedian_cols:
        df['qmedian_' + col] = grouped_quasi_medians(df, train_ixs, groupby=col)
        print col
    return df


def grouped_quasi_medians(df, train_ixs, groupby):
    df['train_ixs'] = train_ixs
    result_grouped = df.groupby(groupby).apply(quasi_medians)
    return np.hstack(result_grouped)


def quasi_medians_old(x):
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
    result = ss.fillna(BASELINE_PRED)
    return result


def quasi_medians(x):
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


def add_datefeats(df):
    tdate = pd.to_datetime(df["transactiondate"])
    df["transactiondate_month"] = tdate.dt.month
    df['transactiondate'] = tdate.dt.quarter
    return df


def add_ranks(df, groupby):
    scores = ['yearbuilt', 'longitude', 'latitude']
    feats = []
    feat_names = []
    for i, score in enumerate(scores):
        rank = rank_by_column(-df[score], df[groupby])
        feats.append(rank)
        feat_names.append('rank_{}_{}'.format(groupby, score))
    for feats in zip(feats, feat_names):
        df[feats[1]] = feats[0]
    return df


def rank_by_column(values, race_id, normalised=True):
    result = np.nan * np.zeros(len(values))
    group_ixs = nptools.get_group_ixs(race_id)
    for group_ix in group_ixs.itervalues():
        race_values = values[group_ix]
        order = race_values.argsort()
        beaten_opponents = order.argsort()
        result[group_ix] = beaten_opponents
        if normalised:
            result[group_ix] = result[group_ix] / float(len(group_ix))
    return result
