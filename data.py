import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from million import tools as nptools
from million._config import test_columns, NULL_VALUE

import seamless as ss

train_cv_ratio = 0.6

FEATS_NOT_TO_USE = ['logerror', 'transactiondate', 'parcelid', 'ParcelId']
df_path = nptools.dropbox() + '/million/cache/full_df.pkl'


def load_data(from_cache=False):
    if from_cache:
        df = ss.io.read_pickle(df_path)
    else:
        print('Loading data...')
        train = pd.read_csv('../input/train_2016_v2.csv')
        train_2017 = pd.read_csv('../input/train_2017.csv')
        prop = pd.read_csv('../input/properties_2016.csv')
        prop2017 = pd.read_csv('../input/properties_2017.csv')
        sample = pd.read_csv('../input/sample_submission.csv')
        prop = _fix_types(prop)
        prop2017 = _fix_types(prop2017)
        df_train2016 = train.merge(prop, how='left', on='parcelid')
        df_train2017 = train_2017.merge(prop2017, how='left', on='parcelid')
        df_train = pd.concat([df_train2016, df_train2017], axis=0)
        sample['parcelid'] = sample['ParcelId']
        df_test = sample.merge(prop2017, how='left', on='parcelid')
        df = create_fulldf(df_train, df_test)
        df = df.fillna(NULL_VALUE)
        df = clean_data(df)
        df = encode_labels(df)
        ss.io.write_pickle(df, df_path)
    return df


def _fix_types(df):
    print('Binding to float32')
    for c, dtype in zip(df.columns, df.dtypes):
            if dtype == np.float64:
                    df[c] = df[c].astype(np.float32)
    return df


def create_fulldf(df_train, df_test):
    df_test['logerror'] = np.nan
    df = df_train.append(df_test, ignore_index=True)
    df = df.drop(test_columns, axis=1)
    return df


def clean_data(df):
    ixs = df['taxdelinquencyflag'] == 'Y'
    df['taxdelinquencyflag'].values[ixs] = 1
    df['taxdelinquencyflag'].values.astype(np.float)
    return df


def select_features(df):
    df = filter_cols(df)
    notnum_cols = df.columns[(df.dtypes == 'O').values]
    print 'notnum_cols', notnum_cols
    all_feats_not_to_use = np.concatenate([notnum_cols, FEATS_NOT_TO_USE, test_columns])
    selected_cols = [c for c in df.columns if c not in all_feats_not_to_use]
    df = df[selected_cols]
    return df


def filter_cols(df):
    not_wanted_feats = ['parcelid', 'logerror', 'transactiondate', 'train_ixs']
    final_cols = [col for col in df.columns if col not in not_wanted_feats]
    return df[final_cols]


def encode_labels(df):
    properties = pd.read_csv('../input/properties_2016.csv', nrows=1)
    for c in properties.columns:
        if df[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))
    return df


def get_lb_ixs(targets):
    test_ixs = targets == NULL_VALUE
    train_ixs = ~test_ixs
    return train_ixs, test_ixs


def get_cv_ixs(df, targets):
    train_ixs, _ = get_lb_ixs(targets)
    train_df = df.iloc[train_ixs]
    new_targets = targets[train_ixs]
    cutoff = int(len(train_df) * train_cv_ratio)
    new_train_ixs = np.array([True]*cutoff + [False]*(len(train_df) - cutoff))
    return train_df, new_targets, new_train_ixs, ~new_train_ixs


def add_month_and_year(df):
    df['transaction_month'] = df['transactiondate'].astype('datetime64').dt.month
    df['transaction_year'] = df['transactiondate'].astype('datetime64').dt.year
    return df


def generate_kaggle_file(sub, file_name):
    print('Writing csv ...')
    sub.to_csv(file_name, index=False, float_format='%.4f')


def generate_simple_kaggle_file(predictions, file_name):
    sub = pd.read_csv('../input/sample_submission.csv')
    for c in sub.columns[sub.columns != 'ParcelId']:
        sub[c] = predictions
    print("\nWriting results to disk ...")
    sub.to_csv(nptools.subs_dir +
            'sub_{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), file_name),
            index=False, float_format='%.5f')
    print("\nFinished ...")
    print(sub.head())

