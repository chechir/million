import numpy as np
from functools import partial
import xgboost as xgb

from million import tools, data
from million._config import NULL_VALUE
from million.experiments.try_stack import n_folds, n_models
from kfuncs import tools as ktools
import seamless as ss
from scipy.optimize import minimize

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
    print score, '*', weights
    return score


def convert_preds_to_list(df):
    result = []
    for col in df.columns:
        result.append(df[col].values)
    return result


if __name__ == '__main__':
    weight = 0.5
    logger = tools.get_logger(LOG_FILE)
    df = data.load_data(from_cache=True)
    df = tools.remove_ouliers(df)
    targets = df['logerror'].values

    train_ixs, test_ixs = data.get_lb_ixs(targets)
    # df = features.add_features(df, train_ixs)

    df = data.select_features(df)
    df_train, train_targets = df.iloc[train_ixs], targets[train_ixs]
    df_test = df.iloc[test_ixs]

    new_train = ss.io.read_pickle(cache_dir + 'ps2_train_2ndx{}_f{}.pkl'.format(n_models, n_folds))
    new_test = ss.io.read_pickle(cache_dir + 'ps2_test_2ndx{}_f{}.pkl'.format(n_models, n_folds))

    params = get_lvl2()
    dtrain = xgb.DMatrix(new_train.values, train_targets)
    dtest = xgb.DMatrix(new_test.values)
    model = xgb.train(
            params,
            dtrain, num_boost_round=300, verbose_eval=2
            )
    preds_train_xgb = model.predict(dtrain)
    preds_test_xgb = model.predict(dtest)
    score = tools.get_mae_loss(train_targets, preds_train_xgb)
    print('train score:{}'.format(score))

    #  ############OPTIM
    init_weights = np.repeat(0.1, n_models)
    all_train_preds = convert_preds_to_list(new_train)
    optim = optimise_weights(
            all_train_preds, train_targets, init_weights, minimise=True)
    print "-", optim.fun
    optimised_weights = optim.x

    for i in range(30):
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

    print 'generating predictions for the test set'
    all_test_preds = convert_preds_to_list(new_test)
    optim_preds = ktools.ensemble_preds(all_test_preds, optimised_weights)

    final_preds = preds_test_xgb * weight + optim_preds*(1 - weight)
    final_preds = final_preds * 1.1
    sub_file_name = 'stkOptim_and_xgbLvl2_x{}_f{}'.format(n_models, n_folds)
    data.generate_simple_kaggle_file(final_preds, sub_file_name)
    score = tools.get_mae_loss(train_targets, final_preds)
    print('train score:{}'.format(score))
    # msg = 'score ens:{}, w:{}.weights, file:{}'.format(score, optimised_weights, sub_file_name)