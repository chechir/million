import numpy as np
from scipy.optimize import minimize
from functools import partial

from million import tools, data
from million._config import NULL_VALUE
from million.experiments.try_stack import n_folds, n_models
from kfuncs import tools as ktools
import seamless as ss


def optimise_weights(preds, targets, init_weights, minimise=True):
    constraints = (
            {'type': 'eq', 'fun': lambda w: 1-sum(w)},
            )
    # weirdness:
    # constraints =(
    #        {'type':'eq','fun':lambda w: 1-sum(w)},
    #        {'type':'eq','fun':lambda w: 0.5 - w[0]}, #weirdness constraint
    #        {'type':'eq','fun':lambda w: 0.17 - w[5]}) #weirdness constraint
    # our weights are bound between 0 and 1
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


cache_dir = tools.cache_dir()
LOG_FILE = tools.experiments() + 'millions_try_stackcomp.txt'
if __name__ == '__main__':
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
    cols = new_train.columns

#    new_train['cat_weird'] = new_train['cat_preds'] + new_train['ker_preds']
#    new_test['cat_weird'] = new_test['cat_preds'] + new_test['ker_preds']
#    new_train['zero'] = np.repeat(0, len(new_train))
#    new_test['zero'] = np.repeat(0, len(new_test))
#    cols = ['cat_weird', 'zero'] + list(cols)

    for col in cols:
        print 'score {}'.format(col),  tools.get_mae_loss(train_targets, new_train[col].values)

    new_train = new_train[cols]
    new_test = new_test[cols]

    init_weights = np.repeat(0.1, n_models)
#    init_weights = [0.1 0.1, 0.1, 0.1, 0.06, 0.08, 0.02, 0.08, 0.42] # (best cv legal)

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
    sub_preds = ktools.ensemble_preds(all_test_preds, optimised_weights)

    sub_file_name = 'stkOptim_x{}_f{}'.format(n_models, n_folds)
    data.generate_simple_kaggle_file(sub_preds, sub_file_name)
    msg = 'score ens:{}, w:{}.weights, file:{}'.format(score, optimised_weights, sub_file_name)
    logger.debug(msg)

