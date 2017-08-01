
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
import random
from datetime import datetime

from million import data, features, tools
from million._config import NULL_VALUE, test_columns, test_dates

# Parameters
XGB_WEIGHT = 0.6840
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0555

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0110   # Baseline based on mean of training data, per Oleg
seed = 7

##### READ IN RAW DATA

def get_lgb_params():
    params = {}
    params['max_bin'] = 10
    params['learning_rate'] = 0.0021 # shrinkage_rate
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'l1'          # or 'mae'
    params['sub_feature'] = 0.5      # feature_fraction -- OK, back to .5, but maybe later increase this
    params['bagging_fraction'] = 0.85 # sub_row
    params['bagging_freq'] = 40
    params['num_leaves'] = 512        # num_leaf
    params['min_data'] = 500         # min_data_in_leaf
    params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
    params['verbose'] = 0
    return params

def get_xgb_params1(y_mean):
    xgb_params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        'base_score': y_mean,
        'silent': 1
    }
    return xgb_params

def get_xgb_params2(y_mean):
    xgb_params = {
        'eta': 0.033,
        'max_depth': 6,
        'subsample': 0.80,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean,
        'silent': 1
    }
    return xgb_params

def get_cols_lgb(df):
    not_wanted_feats = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
            'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag']
    final_cols = [col for col in df.columns if col not in not_wanted_feats]
    return final_cols

def get_outliers_ixs(targets):
    ixs = (targets > -0.4) & (targets < 0.419)
    return ixs

if __name__ == '__main__':
    np.random.seed(seed)
    df_train, df_test = data.load_data(cache=True)
    df = data.create_fulldf(df_train, df_test)

    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)
    #df = features.add_features(df)

    targets = df['logerror'].values
    df = data.select_features(df)

    print df.columns
    df_train, targets, df_test = data.split_data(df, targets)

    ###############
    ################
    ##  LIGHTGBM  ##
    ################
    ################
    cols_lgb = get_cols_lgb(df)
    params = get_lgb_params()
    print("\nFitting LightGBM model ...")
    d_train = lgb.Dataset(df_train[cols_lgb].values, label=targets)

    num_rounds = 430
    clf = lgb.train(params, d_train, num_rounds)
    p_test = clf.predict(df_test[cols_lgb].values)
    print( "\nUnadjusted LightGBM predictions:" )
    print( pd.DataFrame(p_test).head() )

    ################
    ################
    ##  XGBoost   ##
    ################
    ################

    # drop out ouliers
    #df_train=df_train[ df_train.logerror > -0.4 ]
    #df_train=df_train[ df_train.logerror < 0.418 ]
    ixs = ~get_outliers_ixs(targets)

    ##### RUN XGBOOST
    print("\nXGBoost ...")
    y_mean = np.mean(targets)

    dtrain = xgb.DMatrix(df_train.iloc[ixs].values, targets[ixs])
    dtest = xgb.DMatrix(df_test.values)

    num_boost_rounds = 250
    model = xgb.train(get_xgb_params1(y_mean), dtrain, num_boost_round=num_boost_rounds)
    xgb_pred1 = model.predict(dtest)
    print( "\nFirst XGBoost predictions:" )
    print( pd.DataFrame(xgb_pred1).head() )

    ##### RUN XGBOOST AGAIN
    print("\nXGBoost 2 ...")
    num_boost_rounds = 150
    model = xgb.train(get_xgb_params2(y_mean), dtrain, num_boost_round=num_boost_rounds)
    xgb_pred2 = model.predict(dtest)
    print( "\nSecond XGBoost predictions:" )
    print( pd.DataFrame(xgb_pred2).head() )

    xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
    print( "\nCombined XGBoost predictions:" )
    print( pd.DataFrame(xgb_pred).head() )


    ###############
    ################
    ##    OLS     ##
    ################
    ################

    def reg_features(df):
        new_df = features.add_datefeats(df)
        new_df = new_df.fillna(-1.0)
        return new_df

    train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
    properties = pd.read_csv("../input/properties_2016.csv")
    submission = pd.read_csv("../input/sample_submission.csv")
    print(len(train),len(properties),len(submission))

    train = pd.merge(train, properties, how='left', on='parcelid')
    y = train['logerror'].values
    test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
    properties = [] #memory

    exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
    col = [c for c in train.columns if c not in exc]

    train = reg_features(train[col])
    test['transactiondate'] = '2016-01-01' #should use the most common training date
    test = reg_features(test[col])

    reg = ElasticNetCV(normalize=True, l1_ratio=0.8, max_iter=5000)
    reg.fit(train, y); print('fit...')
    print(tools.get_mae_loss(targets, reg.predict(train)))

    ########################
    ########################
    ##  Combine and Save  ##
    ########################
    ########################

    ##### COMBINE PREDICTIONS

    print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
    lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / float((1 - OLS_WEIGHT))
    xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
    baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
    pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test

    print( "\nCombined XGB/LGB/baseline predictions:" )
    print( pd.DataFrame(pred0).head() )

    print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
    for i in range(len(test_dates)):
        test['transactiondate'] = test_dates[i]
        pred = OLS_WEIGHT*reg.predict(reg_features(test)) + (1-OLS_WEIGHT)*pred0
        submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
        print('predict...', i)

    print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
    print( submission.head() )


    ##### WRITE THE RESULTS

    print( "\nWriting results to disk ..." )
    submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

    print( "\nFinished ..." )
