import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error

from million import data, tools
import seamless as ss

TUNING_RESULTS_PATH = ss.paths.experiments() + 'tune_millions_cat_02.json'
y_mean = 0.0102590


def sample_params(random=False):
    if random:
        params = get_random_params()
    else:
        params = get_best_random_params(50)
    return params


def get_random_params():
    cat_params = {
        'iterations': 1900,
        'thread_count': 2,
        'loss_function': 'MAE',
        'auto_stop_pval': 0.05,
        'od-type': 'Iter',
        'iterations_wait': 40,
        'learning_rate': 0.02,
        'depth': np.random.randint(4, 9),
        'l2_leaf_reg': np.random.randint(1, 16),
        'rsm': np.random.uniform(0.1, 1.),
        'bagging_temperature': np.random.uniform(0.1, 1),
        'fold_permutation_block_size': np.random.randint(1, 5),
        'gradient_iterations': 1,
        'random_seed': np.random.randint(0, 9000),
        'has_time': bool(np.random.choice([0, 1])),
        'use_best_model': True,
        'verbose': True,
        'name': 'experiment',
    }
    return cat_params


def get_best_random_params(num_elements):
    random_data = [get_random_params() for _ in range(num_elements)]
    random_df = pd.DataFrame(random_data)
    json_df = tools.read_special_json(TUNING_RESULTS_PATH)
    columns = [
            'learning_rate', 'depth', 'l2_leaf_reg', 'rsm', 'bagging_temperature',
            'fold_permutation_block_size', 'gradient_iterations', 'has_time', 'learning_rate'
            ]
    mm_train = json_df[columns]
    targets = json_df['mae']
    model = RFR(n_estimators=50)
    model.fit(mm_train, targets)
    predicted_losses = model.predict(random_df[columns])
    best_prediction_ix = np.argmin(predicted_losses)
    result = random_df.iloc[best_prediction_ix]
    result = result.to_dict()
    result['has_time'] = bool(result['has_time'])
    result['verbose'] = bool(result['verbose'])
    result['use_best_model'] = bool(result['use_best_model'])
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
    # eval_set = [(df_test.values, test_targets)]
    eval_set = [df_test.values, test_targets]

    while True:
        params = sample_params(random=False)
        # params = model_params.get_ctune729c()
        print params
        model = CatBoostRegressor(**params)
        model.fit(df_train.values, train_targets, eval_set=eval_set)

        predictions = model.predict(df_test)
        mae = tools.get_mae_loss(test_targets, predictions)
        mse = mean_squared_error(test_targets, predictions)
        losses = {'mse': mse, 'mae': mae}
        params = dict(params, **losses)
        tools.write_results_to_json(params, TUNING_RESULTS_PATH)
