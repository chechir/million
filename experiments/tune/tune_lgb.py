import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error

from million import data, tools
import seamless as ss

TUNING_RESULTS_PATH = ss.paths.experiments() + 'tune_millions_lgb2.json'


def sample_params(random=False):
    if random:
        params = get_random_params()
    else:
        params = get_best_random_params(50)
    return params


def get_random_params():
    params = {}
    params['learning_rate'] = 0.01
    params['min_data'] = np.random.randint(50, 8000)         # min_data_in_leaf
    params['min_hessian'] = np.random.uniform(0., 0.6)     # min_sum_hessian_in_leaf
    params['lambda_l1'] = np.random.uniform(0, 5.)     # l1 regularization
    params['lambda_l2'] = np.random.uniform(0, 5.)     # l2 regularization
    params['num_leaves'] = np.random.randint(10, 413)        # num_leaf
    params['subsample_for_bin'] = np.random.choice(np.round((np.exp(0.2*np.arange(50, 80)))))
    params['min_child_samples'] = np.random.uniform(3, 100)     # l2 regularization
    params['max_depth'] = np.random.choice(np.append(-1, np.random.randint(5, 20)))
    params['min_child_weight'] = np.random.randint(1, 130)
    params['subsample_freq'] = np.random.randint(1, 10)
    params['subsample'] = np.random.uniform(0.3, 1)
    params['colsample_bytree'] = np.random.uniform(0.3, 1)
    params['seed'] = np.random.randint(1, 9999999)
    params['min_split_gain'] = 0
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'regression'
    params['metric'] = 'mae'          # or 'mae'
    params['n_estimators'] = 999999
    params['num_threads'] = 2
    return params


def get_best_random_params(num_elements):
    random_data = [get_random_params() for _ in range(num_elements)]
    random_df = pd.DataFrame(random_data)
    json_df = tools.read_special_json(TUNING_RESULTS_PATH)
    columns = [
            'min_data', 'min_hessian', 'lambda_l1', 'lambda_l2',
            'num_leaves', 'subsample_for_bin', 'min_child_samples', 'max_depth',
            'min_child_weight', 'subsample_freq', 'subsample', 'colsample_bytree'
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
    df = data.load_data(from_cache=True)
    df = tools.remove_ouliers(df)
    targets = df['logerror'].values

    df, targets, train_ixs, test_ixs = data.get_cv_ixs(df, targets)
    df = data.select_features(df)
    df_train, train_targets = df.iloc[train_ixs], targets[train_ixs]

    df_test, test_targets = df.iloc[test_ixs], targets[test_ixs]
    eval_set = [(df_test.values, test_targets)]

    while True:
        params = sample_params(random=True)
        model = LGBMRegressor(**params)
        model.fit(
                df_train, train_targets,
                eval_set=[(df_test, test_targets)],
                early_stopping_rounds=50
                )
        predictions = model.predict(df_test)
        mae = tools.get_mae_loss(test_targets, predictions)
        mse = mean_squared_error(test_targets, predictions)
        losses = {'mse': mse, 'mae': mae}
        params = dict(params, **losses)
        tools.write_results_to_json(params, TUNING_RESULTS_PATH)

