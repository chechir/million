from sklearn.linear_model import ElasticNetCV
from million import data
from million._config import NULL_VALUE, test_columns, test_dates
from million import model_params
seed = 147
n_bags = 5

def delete_some_outliers(df, targets):
    outlier_ixs = (targets > 0.419) | (targets < -0.4)
    filter_ixs = np.array([(np.random.normal() > 0.5) & o for o in outlier_ixs])
    return df.iloc[~filter_ixs], targets[~filter_ixs]

# Parameters
XGB_WEIGHT = 0.6750
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0550

XGB1_WEIGHT = 0.85  # Weight of first in combination of two XGB models

BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg

# Version 26:
# Getting rid of the LightGBM validation, since this script doesn't use the result.
# Now use all training data to fit model.
# I have a separate script for validation:
#    https://www.kaggle.com/aharless/lightgbm-outliers-remaining-cv


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import gc
import random
import datetime as dt


##### READ IN RAW DATA

print( "\nReading data from disk ...")
prop = pd.read_csv('../input/properties_2016.csv')
train = pd.read_csv("../input/train_2016_v2.csv")


################
################
##  LightGBM  ##
################
################

# This section is (I think) originally derived from SIDHARTH's script:
#   https://www.kaggle.com/sidharthkumar/trying-lightgbm
# which was forked and tuned by Yuqing Xue:
#   https://www.kaggle.com/yuqingxue/lightgbm-85-97
# and updated by me (Andy Harless):
#   https://www.kaggle.com/aharless/lightgbm-with-outliers-remaining
# and a lot of additional changes have happened since then,
#   the most recent of which are documented in my comments above

##### PROCESS DATA FOR LIGHTGBM

print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
#x_train['Ratio_1'] = x_train['taxvaluedollarcnt']/x_train['taxamount']
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)



##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction -- OK, back to .5, but maybe later increase this
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()
del x_train; gc.collect()

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")
sample = pd.read_csv('../input/sample_submission.csv')
print("   ...")
sample['parcelid'] = sample['ParcelId']
print("   Merge with property data ...")
df_test = sample.merge(prop, on='parcelid', how='left')
print("   ...")
del sample, prop; gc.collect()
print("   ...")
#df_test['Ratio_1'] = df_test['taxvaluedollarcnt']/df_test['taxamount']
x_test = df_test[train_columns]
print("   ...")
del df_test; gc.collect()
print("   Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
print("   ...")
x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")
p_test = clf.predict(x_test)

del x_test; gc.collect()

print( "\nUnadjusted LightGBM predictions:" )
print( pd.DataFrame(p_test).head() )




################
################
##  XGBoost   ##
################
################

# This section is (I think) originally derived from Infinite Wing's script:
#   https://www.kaggle.com/infinitewing/xgboost-without-outliers-lb-0-06463
# inspired by this thread:
#   https://www.kaggle.com/c/zillow-prize-1/discussion/33710
# but the code has gone through a lot of changes since then


##### RE-READ PROPERTIES FILE
##### (I tried keeping a copy, but the program crashed.)

print( "\nRe-reading properties file ...")
properties = pd.read_csv('../input/properties_2016.csv')



##### PROCESS DATA FOR XGBOOST

print( "\nProcessing data for XGBoost ...")
for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape        
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')     
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))


##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'seed' : 123456,
    'alpha': 0.4, 
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 250
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
xgb_pred1 = model.predict(dtest)

print( "\nFirst XGBoost predictions:" )
print( pd.DataFrame(xgb_pred1).head() )



##### RUN XGBOOST AGAIN

###training full:
np.random.seed(seed)
df_train, df_test = data.load_data(cache=True)
df = data.create_fulldf(df_train, df_test)

df = df.fillna(NULL_VALUE)
df = data.clean_data(df)
df = data.encode_labels(df)
#df = features.add_features(df)

logerror = df['logerror'].values
targets = logerror
df = data.select_features(df)

df_train, targets, df_test = data.split_data(df, logerror)

dtest = xgb.DMatrix(df_test.values)
dtrain = xgb.DMatrix(df_train.values, targets)
params = model_params.get_xtune11k()

sub_preds = np.repeat(0, len(df_test))
num_boost_rounds = 110
for i in range(n_bags):
    model = xgb.train(
            params,
            dtrain, num_boost_round=num_boost_rounds,
            )
    sub_preds = model.predict(dtest) + sub_preds

    #prepare for the next iteration
    df_bag, bag_targets = delete_some_outliers(df_train, targets)
    dtrain = xgb.DMatrix(df_bag.values, bag_targets)
    print(i, df_bag.shape)
    num_boost_rounds = 155
    #params['seed'] = i
sub_preds = sub_preds / n_bags


##### COMBINE XGBOOST RESULTS

xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*sub_preds
#xgb_pred = xgb_pred1

print( "\nCombined XGBoost predictions:" )
print( pd.DataFrame(xgb_pred).head() )

del train_df
del x_train
del x_test
del properties
del dtest
del dtrain
del xgb_pred1
gc.collect()



################
################
##    OLS     ##
################
################

# This section is derived from the1owl's notebook:
#    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
# which I (Andy Harless) updated and made into a script:
#    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols

np.random.seed(17)
random.seed(17)

train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
properties = pd.read_csv("../input/properties_2016.csv")
submission = pd.read_csv("../input/sample_submission.csv")
print(len(train),len(properties),len(submission))

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    #df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / float(len(y))

train = pd.merge(train, properties, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')
properties = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])

reg = ElasticNetCV(normalize=True, l1_ratio=0.8, max_iter=5000)
reg.fit(train, y); print('fit...')
print(MAE(y, reg.predict(train)))
train = [];  y = [] #memory


########################
########################
##  Combine and Save  ##
########################
########################


##### COMBINE PREDICTIONS

print( "\nCombining XGBoost, LightGBM, and baseline predicitons ..." )
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / float(1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / float(1 - OLS_WEIGHT)
baseline_weight0 =  BASELINE_WEIGHT / float(1 - OLS_WEIGHT)
pred0 = xgb_weight0*xgb_pred + baseline_weight0*BASELINE_PRED + lgb_weight*p_test

print( "\nCombined XGB/LGB/baseline predictions:" )
print( pd.DataFrame(pred0).head() )

print( "\nPredicting with OLS and combining with XGB/LGB/baseline predicitons: ..." )
for i in range(len(test_dates)):
    test['transactiondate'] = test_dates[i]
    pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0
    submission[test_columns[i]] = [float(format(x, '.5f')) for x in pred]
    print('predict...', i)

print( "\nCombined XGB/LGB/baseline/OLS predictions:" )
print( submission.head() )



##### WRITE THE RESULTS

from datetime import datetime

print( "\nWriting results to disk ..." )
submission.to_csv('sub_kaggle_plus_bagged{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print( "\nFinished ...")


########################
########################
##  Version Comments  ##
########################
########################

#... version 42, continued from the top

# version 44
#    Confirmed that submitting the same exact script in immediate succession
#      produces the same result.

# version 45
#    Now try it with this comment added and see if it's still the same (and yes it is!).

# version 46
#    Add an extra XGB run with different parameters.  (LB score gets worse, but
#      perhaps because LightGBM results are different, even though I changed nothing
#        up to the point in the script where it finishes running LightGBM.  This is annoying.)

# version 47
#    Comment out version 46 changes.  But results will be completely new 
#      because of black magic perpetrated by the LightGBM imps.

# version 54
#    Try higher XGB_WEIGHT (in this case .6900) per danieleewww -> LB .0644060

# version 56
#    Try XGB_WEIGHT=.71 -> LB .0644069

# version 57
#    Try XGB_WEIGHT=.67 -> LB .0644062
#      Quadratic approximation would imply optimum of about .684,
#         which I will note but not worth wasting another submission on it.
#      (Subsequent changes will alter the optimal weight anyhow.)

# version 58
#    Try bringing back the second XGB fit, but with a smaller weight (0.15 vs 0.40)
#      -> LB .0644053

# version 59
#    Try XGB_WEIGHT=.70 -> LB .0644056
#    Seems unlikely that this would reduce the LB score after the 2nd XGB model improved it.
#      I suspect that the LIghtGBM imps are up to their old tricks.

# version 60
#    Try XGB_WEIGHT=.684, per earlier approximated optimum
#    Also, I'm putting XGB1_WEIGHT at the top, although this will tempt
#       all sort of evil mischeif from the LightGBM imps
#    LB .0644052

# version 61
#    Try XGB1_WEIGHT=.80 -> LB .0644049

# version 62
#    Try XGB1_WEIGHT=.75 -> LB .0644055

# version 63
#    Change XGB1_WEIGHT to .8083 (quadratic approximated optimum)
#    Add ratio of taxvaluedollarcnt/taxamount (to LGB), per RpyGamer suggesiton
#    LB .0644238

# version 65
#    Get rid of tax ratio.  (Maybe try it later in XGB?)
