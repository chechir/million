import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error

from million import data, features, tools
from million._config import NULL_VALUE, test_columns, test_dates
import seamless as ss

TUNING_RESULTS_PATH = ss.paths.experiments() + 'tune_millions_xgb.json'
y_mean = 0.0102590

def sample_params(random=False):
    if random:
        params = get_random_params()
    else:
        params = get_best_random_params(50)
    return params

def get_random_params():
    xgb_params = {
        #"objective": "binary:logistic",
        #"eval_metric": 'logloss', 
        'eval_metric': 'mae',
        "eta": 0.018,
        "max_depth": np.random.randint(2, 14),
        "min_child_weight":np.random.choice(np.arange(100)),
        "alpha": np.random.choice([0, np.random.uniform(0, 7)]),
        "colsample_bytree": np.random.uniform(0.2, 1.),
        "subsample": np.random.uniform(0.4, 1.),
        "lambda":np.random.choice([np.random.uniform(0, 8)]),
        'seed': np.random.randint(0, 1000),
        "silent": 1,
        'nthread':4,
        'base_score': y_mean,
    }
    return xgb_params


def get_best_random_params(num_elements):
    random_data = [get_random_params() for _ in range(num_elements)]
    random_df = pd.DataFrame(random_data)
    json_df = tools.read_special_json(TUNING_RESULTS_PATH)
    columns = [
            'max_depth', 'min_child_weight', 'alpha', 'colsample_bytree', 'subsample',
            'subsample', 'lambda',
            ]
    mm_train = json_df[columns]
    targets = json_df['mae']
    model = RFR(n_estimators=50)
    model.fit(mm_train, targets)
    predicted_losses = model.predict(random_df[columns])
    best_prediction_ix = np.argmin(predicted_losses)
    result = random_df.iloc[best_prediction_ix]
    result = result.to_dict()
    print 'Expected Loss: {}'.format(predicted_losses[best_prediction_ix])
    return result

if __name__ == '__main__':

    df_train, df_test = data.load_data(cache=True)
    df = data.create_fulldf(df_train, df_test)

    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)
    #df = features.add_features(df)

    targets = df['logerror'].values
    df = data.select_features(df)

    print df.columns
    df_full_train, targets, df_test = data.split_data(df, targets)
    df_train, df_test, train_targets, test_targets = data.split_cv(df_full_train, targets, 0.8)

    dtrain = xgb.DMatrix(df_train.values, train_targets)
    dtest = xgb.DMatrix(df_test.values, test_targets)

    y_mean = np.mean(train_targets)
    watchlist  = [(dtrain,'train'),(dtest,'eval')]

    num_boost_rounds = 80000
    while True:
        params = sample_params(random=False)
        model = xgb.train(
                params,
                dtrain, num_boost_round=num_boost_rounds,
                evals=watchlist,
                early_stopping_rounds=15
                )
        predictions = model.predict(dtest)
        mae = tools.get_mae_loss(test_targets, predictions)
        mse = mean_squared_error(test_targets, predictions)
        losses = {'mse':mse, 'mae':mae}
        params = dict(params, **losses)
        tools.write_results_to_json(params, TUNING_RESULTS_PATH)

