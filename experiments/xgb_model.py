import xgboost as xgb
import pandas as pd
import numpy as np
from million._config import NULL_VALUE, cv_split
from million import data, features

num_rounds = 10000

def _get_xgb_params():
    params = {
            "objective": 'reg:linear',
            "booster": 'gbtree',
            "eval_metric": 'rmse',
            'seed':523,
            'silent': 1,
            "max_depth": 4,
            'eta': 0.018,
            "subsample": 0.95,
            "alpha": 4.15,
            "lambda": 5.86,
            "colsample_bytree": 0.70,
            'min_child_weight': 72,
            'verbose_eval' : 50,
            }
    return params

def train_xgb_cv(df_train, targets):
    params = _get_xgb_params()
    x_train, y_train, x_valid, y_valid = df_train[:cv_split], targets[:cv_split], df_train[cv_split:], targets[cv_split:]
    d_train = xgb.DMatrix(x_train.values, label=y_train)
    d_valid = xgb.DMatrix(x_valid.values, label=y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    print('Training ...')
    model = xgb.train(params, d_train, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=10)
    return model

def predict_xgb(model, xdata):
    d_test = xgb.DMatrix(xdata)
    predictions = model.predict(d_test)
    return predictions

def predict_multiple_months(df_test, predict_func):
    sub = pd.read_csv('../input/sample_submission.csv')
    df = df_test.copy()
    for c in sub.columns[sub.columns != 'ParcelId']:
        df['transaction_month'] = np.repeat(c[4:6], len(df))
        df['transaction_year'] = np.repeat(c[0:4], len(df))
        predictions = predict_func(model, df.values)
        print 'predicting for ' + c + ' ' + str(predictions.sum())
        sub[c] = predictions
    return sub

if __name__ == '__main__':
    df_train, df_test = data.load_data()
    1/0
    df = data.create_fulldf(df_train, df_test)
    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)
    df = features.add_features(df)
    #df = data.add_month_and_year(df)
    targets = df['logerror'].values
    df = data.select_features(df)
    df_train, targets, df_test = data.split_data(df, targets)
    model = train_xgb_cv(df_train, targets)

    sub = predict_multiple_months(df_test, predict_xgb)
    data.generate_kaggle_file(sub, 'sub/xgb_try_exper_quasi.csv')

