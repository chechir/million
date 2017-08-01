import numpy as np
import lightgbm as lgb
from million import data

def train_lgb_cv(xtrain, targets):
    cv_split = 90000
    params = _get_lgb_params()
    x_train, y_train, x_valid, y_valid = xtrain[:cv_split], targets[:cv_split], xtrain[cv_split:], targets[cv_split:]
    x_train = x_train.astype(np.float32, copy=False)
    x_valid = x_valid.astype(np.float32, copy=False)

    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    watchlist = [d_valid]
    model = lgb.train(params, d_train, 490, watchlist)
    print('Training ...')
    return model

def _get_lgb_params():
    params = {}
    params['max_bin'] = 10
    params['learning_rate'] = 0.0021 # shrinkage_rate
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'l2'          # or 'mae'
    params['sub_feature'] = 0.5      # feature_fraction 
    params['bagging_fraction'] = 0.85 # sub_row
    params['bagging_freq'] = 40
    params['num_leaves'] = 60        # num_leaf
    params['min_data'] = 500         # min_data_in_leaf
    params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
    return params

def predict_lgb(model, xdata):
    print("Start prediction ...")
    # num_threads > 1 will predict very slow in kernal
    model.reset_parameter({"num_threads":1})
    predictions = model.predict(xdata)
    predictions = 0.98*predictions + 0.02*0.011
    return predictions

if __name__ == '__main__':
    df_train, df_test = data.load_data(cache=False)
    df = data.create_fulldf(df_train, df_test)
    df = df.fillna(data.NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)
#    df = features.add_features(df)
    targets = df['logerror'].values
    df = data.select_features(df)
    print df.columns
    df_train, targets, df_test = data.split_data(df, targets)

    model = train_lgb_cv(df_train.values, targets)
    predictions = predict_lgb(model, df_test.values)
    data.generate_simple_kaggle_file(predictions, 'sub/lgb_try_simple_params.csv')

