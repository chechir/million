from million._config import NULL_VALUE
import numpy as np
import os
import logging


def remove_ouliers(df):
    low_outlier_ixs = (df['logerror'].values < -0.4) & (df['logerror'].values != NULL_VALUE)
    high_outlier_ixs = df['logerror'].values > 0.41
    outlier_ixs = low_outlier_ixs | high_outlier_ixs
    result_df = df.iloc[~outlier_ixs]
    return result_df


def normalise_data(train, test):
    for col_ix in range(train.shape[1]):
        mean_col = train[:, col_ix].astype(np.float).mean()
        std_col = train[:, col_ix].astype(np.float).std()
        train[:, col_ix] = (train[:, col_ix].astype(np.float) - mean_col)/std_col
        test[:, col_ix] = (test[:, col_ix].astype(np.float) - mean_col)/std_col
    return train, test


def delete_some_outliers(df, targets):
    outlier_ixs = (targets > 0.419) | (targets < -0.4)
    filter_ixs = np.array([(np.random.normal() > 0.5) & o for o in outlier_ixs])
    return df.iloc[~filter_ixs], targets[~filter_ixs]


def ensemble_preds(predictions, weights):
    return np.average(predictions, weights=weights, axis=0)


def get_mae_loss(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / float(len(y))


def ix_to_bool(ix, length):
    boolean_mask = np.repeat(False, length)
    boolean_mask[ix] = True
    return boolean_mask


def get_logger(file_name):
    logger_format = '%(asctime)s - %(filename)s - %(message)s'
    logging.basicConfig(filename=file_name, level=logging.DEBUG, format=logger_format)
    logger = logging.getLogger()
    return logger


def dropbox():
    path = os.path.expanduser('~/Dropbox/')
    if not os.path.isdir(path):
        path = '/Dropbox/'
    return path


def cache_dir():
    path = dropbox() + '/million/cache/'
    return path


def experiments():
    dropbox_path = dropbox()
    path = dropbox_path + 'experiments/'
    return path


def convert_to_python_types(dic):
    for key in dic:
        dic[key] = dic[key].item()
    return dic


subs_dir = dropbox() + 'million/sub/'
