import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from million import tools as nptools
from million._config import test_columns

FEATS_NOT_TO_USE = [ 'logerror', 'transactiondate', 'parcelid', 'ParcelId', ]
df_train_path = nptools.dropbox() + '/million/cache/train.pkl'
df_test_path = nptools.dropbox() + '/million/cache/test.pkl'

def load_data(cache=False):
    if cache:
        df_train = nptools.read_pickle(df_train_path)
        df_test = nptools.read_pickle(df_test_path)
    else:
        train = pd.read_csv('../input/train_2016_v2.csv')
        prop = pd.read_csv('../input/properties_2016.csv')
        sample = pd.read_csv('../input/sample_submission.csv')
        prop = _fix_types(prop)
        print prop.shape
        df_train = train.merge(prop, how='left', on='parcelid')
        sample['parcelid'] = sample['ParcelId']
        df_test = sample.merge(prop, how='left', on='parcelid')
        #import pdb; pdb.set_trace()
        nptools.write_pickle(df_train, df_train_path)
        nptools.write_pickle(df_test, df_test_path)
    return df_train, df_test

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
    not_wanted_feats = ['parcelid', 'logerror', 'transactiondate']
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

def split_data(df, targets):
    ixs = nptools.get_test_ixs(targets)
    df_train = df.iloc[~ixs]
    df_test = df.iloc[ixs]
    targets = targets[~ixs]
    return df_train, targets, df_test

def split_cv(df, targets, ratio):
    cutoff = int(len(df) * ratio)
    train_df = df[0:cutoff]
    val_df = df[cutoff:]
    train_targets = targets[0:cutoff]
    val_targets = targets[cutoff:]
    return train_df, val_df, train_targets, val_targets

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
    print( "\nWriting results to disk ..." )
    sub.to_csv('sub_{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), file_name),
            index=False, float_format='%.5f')
    print( "\nFinished ..." )
    print( sub.head() )

