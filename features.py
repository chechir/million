from million import tools as nptools
import numpy as np
import pandas as pd
from million.tools import get_test_ixs

def add_features(df):
    #df['cnt_censustractandblock'] = column_count(df, 'censustractandblock')
    #df['cnt_parcelid'] = column_count(df, df['parcelid'].values)
    #df['cnt_regionidcity'] = column_count(df, 'regionidcity')
    #df['cnt_regionidzip'] = column_count(df, 'regionidzip')
    #df['cnt_propertylandusetypeid'] = column_count(df, 'propertylandusetypeid')
    #df['cnt_propertycountylandusecode'] = column_count(df, 'propertycountylandusecode')
    cnt_cols = ['propertylandusetypeid']
    for col in cnt_cols:
        df['cnt_'+ col] = column_count(df, col)

    #df = add_ranks(df, 'regionidzip') #df = add_ranks(df, 'propertycountylandusecode')
    #df = add_datefeats(df)

    return df

def add_datefeats(df):
    tdate = pd.to_datetime(df["transactiondate"])
    df["transactiondate_month"] = tdate.dt.month
    df['transactiondate'] = tdate.dt.quarter
    return df

def get_train_df(df):
    test_ixs = get_test_ixs(df['logerror'].values)
    train_df = df.iloc[~test_ixs]
    return train_df

def column_count(df, col):
    train_df = get_train_df(df)
    group_ixs = nptools.get_group_ixs(train_df[col].values)
    result = np.nan * np.zeros(len(df))
    for key, group_ix in group_ixs.iteritems():
        result[df[col]==key] = len(group_ix)
    return result

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

def add_quasi_ratios(train_df, val_df, targets):
    train=train_df.copy()
    val=val_df.copy()
    train["id"] = np.arange(train.shape[0])
    val["id"] = np.arange(val.shape[0])

    for col in ['regionidzip', 'regionidneighborhood']:
        feat_train, feat_test = calc_quasi_targets(train, val, targets, col)
        train_df['quasi_{}'.format(col)] = feat_train
        val_df['quasi_{}'.format(col)] = feat_test
    return train_df, val_df

def calc_quasi_targets(train, val, Y_train, groupby):
    train['outcome'] = Y_train
    #val['players_key']=val["player_id"]

    train_stats = train[[groupby, 'outcome']].groupby(groupby).agg(['sum', 'count'])
    train_stats.columns = train_stats.columns.droplevel(0)
    train_stats2 = train.merge(train_stats, how='left', left_on=groupby, right_index=True)[["id",groupby, "sum", "count", "outcome"]]
    train_stats2["players_avg"] = (train_stats2["sum"] - train_stats2["outcome"]) / (train_stats2["count"] - 1)
    train_stats2["players_avg"].fillna(-1, inplace=True)

    players_win_stat_val = val.merge(train_stats, how='left', left_on=groupby, right_index=True)[["id",groupby, "sum", "count"]]
    players_win_stat_val["players_avg"] = (players_win_stat_val["sum"]) / (players_win_stat_val["count"])
    players_win_stat_val["players_avg"].fillna(-1, inplace=True)

    train=train.merge(train_stats2[["id", "players_avg"]], how="left", left_on="id", right_on="id")
    val=val.merge(players_win_stat_val[["id", "players_avg"]], how="left", left_on="id", right_on="id")
    return(train, val)

