import numpy as np
from functools import partial
import xgboost as xgb

seed = 1
np.random.seed(seed)

import pandas as pd
from million import tools, data, model_params
from million._config import NULL_VALUE
from million.experiments.try_stack import n_folds, n_models as n_models1
from million.experiments.try_stack2 import n_models as n_models2
from kfuncs import tools as ktools
import seamless as ss
from scipy.optimize import minimize

n_models = n_models1 + n_models2
cache_dir = tools.cache_dir()
LOG_FILE = tools.experiments() + 'millions_try_stackcomp.txt'


def get_lvl2():
    xgb_params = {
        'objective': 'reg:linear',
        'gamma': 0.0001,
        'eta': 0.1,
        'max_depth': 2,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'eval_metric': 'mae',
        'seed': 777,
        'base_score': 0.01026,
        'silent': 1
    }
    return xgb_params


def optimise_weights(preds, targets, init_weights, minimise=True):
    constraints = (
            {'type': 'eq', 'fun': lambda w: 1-sum(w)},
            )
    bounds = [(-1, 1)]*len(preds)
    func = partial(optim_func, preds=preds, targets=targets)
    result = minimize(
            func, init_weights, method='SLSQP',
            bounds=bounds, constraints=constraints)
    return result


def optim_func(weights, preds, targets):
    final_prediction = ktools.ensemble_preds(preds, weights)
    score = 1000000*tools.get_mae_loss(final_prediction, targets)
    return score


def convert_preds_to_list(df):
    result = []
    for col in df.columns:
        result.append(df[col].values)
    return result


def get_new_guys():
    train = ss.io.read_pickle(cache_dir + 'ps2_train_2ndx{}_f{}.pkl'.format(n_models1, n_folds))
    test = ss.io.read_pickle(cache_dir + 'ps2_test_2ndx{}_f{}.pkl'.format(n_models1, n_folds))
    train2 = ss.io.read_pickle(cache_dir + 'ps2_train2_2ndx{}_f{}.pkl'.format(n_models2, n_folds))
    test2 = ss.io.read_pickle(cache_dir + 'ps2_test2_2ndx{}_f{}.pkl'.format(n_models2, n_folds))
    return pd.concat([train, train2], axis=1), pd.concat([test, test2], axis=1)


if __name__ == '__main__':
    logger = tools.get_logger(LOG_FILE)
    df = data.load_data(from_cache=True)
    df = tools.remove_ouliers(df)
    targets = df['logerror'].values
    train_ixs, test_ixs = data.get_lb_ixs(targets)
    df = data.select_features(df)
    df_train, train_targets = df.iloc[train_ixs], targets[train_ixs]
    df_test = df.iloc[test_ixs]

    new_train, new_test = get_new_guys()

    print('XGBoost... ')
    params = model_params.get_lvl2()
    dtrain = xgb.DMatrix(new_train.values, train_targets)
    dtest = xgb.DMatrix(new_test.values)
    preds_train_xgb = np.zeros(len(new_train))
    preds_test_xgb = np.zeros(len(new_test))
    n_bags = 7
    for i in range(n_bags):
        model = xgb.train(
                params,
                dtrain, num_boost_round=300, verbose_eval=2
                )
        preds_train_xgb += model.predict(dtrain)
        preds_test_xgb += model.predict(dtest)
    preds_train_xgb /= n_bags
    preds_test_xgb /= n_bags
    score = tools.get_mae_loss(train_targets, preds_train_xgb)
    print('train score:{}'.format(score))

    #  ############Keras
    print('nnet... ')
    x_train = new_train.values
    x_test = new_test.values
    model = model_params.get_lvl2nn(x_train.shape[1])
    batch_size = 256
    epochs = 10
    history = model.fit(
            x_train, train_targets,
            nb_epoch=epochs, batch_size=batch_size)
    model.history = history
    preds_train_nn = model.predict(x_train).squeeze()
    preds_test_nn = model.predict(x_test).squeeze()
    score = tools.get_mae_loss(train_targets, preds_train_nn)
    print('train score:{}'.format(score))

    #  ############OPTIM
    print('Optim... ')
    init_weights = np.repeat(0.1, n_models)
    all_train_preds = convert_preds_to_list(new_train)
    optim = optimise_weights(
            all_train_preds, train_targets, init_weights, minimise=True)
    print "-", optim.fun
    optimised_weights = optim.x

    for i in range(40):
        optim = optimise_weights(
                all_train_preds, train_targets, optimised_weights, minimise=True)
        optimised_weights = optim.x
        print "-", optim.fun

    train_preds = ktools.ensemble_preds(all_train_preds, init_weights)
    score = tools.get_mae_loss(train_targets, train_preds)
    print 'score ens manual:', score, init_weights

    train_preds = ktools.ensemble_preds(all_train_preds, optimised_weights)
    score = tools.get_mae_loss(train_targets, train_preds)
    print 'score ens with optim:', score, optimised_weights

    all_test_preds = convert_preds_to_list(new_test)
    optim_preds = ktools.ensemble_preds(all_test_preds, optimised_weights)

    weights = [0.45, 0.35, 0.20]
    print 'generating predictions for the test set. weiths:{}'.format(weights)
    final_preds = tools.ensemble_preds([preds_test_xgb, optim_preds, preds_test_nn], weights)
    final_preds = final_preds * 1.1

    final_preds_train = tools.ensemble_preds([preds_train_xgb, train_preds, preds_train_nn], weights)
    score = tools.get_mae_loss(train_targets, final_preds_train)
    print('train score:{}'.format(score))

    sub_file_name = 'stk_3lvl2_models_x{}_f{}'.format(n_models, n_folds)
    data.generate_simple_kaggle_file(final_preds, sub_file_name)
