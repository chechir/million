import numpy as np
from scipy.optimize import minimize
from functools import partial

from million import tools, data
from million._config import NULL_VALUE
from million.experiments.try_stack import n_folds, n_models
from kfuncs import tools as ktools

def optimise_weights(preds, targets, init_weights, minimise=True):
    constraints =(
            {'type':'eq','fun':lambda w: 1-sum(w)},
            )
    #weirdness:
    #constraints =(
    #        {'type':'eq','fun':lambda w: 1-sum(w)},
    #        {'type':'eq','fun':lambda w: 0.5 - w[0]}, #weirdness constraint
    #        {'type':'eq','fun':lambda w: 0.17 - w[5]}) #weirdness constraint
    #our weights are bound between 0 and 1
    bounds = [(-1,1)]*len(preds)
    func = partial(optim_func, preds=preds, targets=targets)
    result = minimize(
            func, init_weights, method='SLSQP',
            bounds=bounds, constraints=constraints)
    #return result['x'], result['fun']
    print result['fun']
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
    df_train, df_test = data.load_data(cache=True)
    df = data.create_fulldf(df_train, df_test)

    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)

    logerror = df['logerror'].values
    targets = logerror
    df = data.select_features(df)

    df_train, targets, df_test = data.split_data(df, logerror)

    new_train = tools.read_pickle(cache_dir + 'ps_train_2ndx{}_f{}.pkl'.format(n_models, n_folds))
    new_test = tools.read_pickle(cache_dir + 'ps_test_2ndx{}_f{}.pkl'.format(n_models, n_folds))
    cols = new_train.columns

    #new_train0 = tools.read_pickle(cache_dir + 'ps_train_2ndx{}_f{}.pkl'.format(5, n_folds))
    #new_test0 = tools.read_pickle(cache_dir + 'ps_test_2ndx{}_f{}.pkl'.format(5, n_folds))
    new_train['cat_weird'] = new_train['cat_preds'] + new_train['ker_preds']
    new_test['cat_weird'] = new_test['cat_preds'] + new_test['ker_preds']
    new_train['zero'] = np.repeat(0, len(new_train))
    new_test['zero'] = np.repeat(0, len(new_test))
    cols = ['cat_weird', 'zero'] + list(cols)

    for col in cols:
        print 'score {}'.format(col),  tools.get_mae_loss(targets, new_train[col])

    new_train = new_train[cols]
    new_test = new_test[cols]

    #init_weights = [0.00005, 0.0001, .02, .07, 0.18, 0.07, 0.06, 0.03, 0.05]
    init_weights = [0., 0.01, 0.34, 0.02, 0.06, 0.08, 0.02, 0.08, 0.42] # (best cv legal)
    #init_weights = [0.5, 0.01, .02, .07, 0.18, 0.07, 0.06, 0.03, 0.05] # (best werid)
    #init_weights = [0.5, 0.01, .01, .01, 0.11, 0.01, 0.01, 0.01, 0.01]

    all_train_preds = convert_preds_to_list(new_train)
    optim = optimise_weights(
            all_train_preds, targets, init_weights, minimise=True)
    print "-", optim.fun
    optimised_weights = optim.x

    for i in range(10):
        optim = optimise_weights(
                all_train_preds, targets, optimised_weights, minimise=True)
        optimised_weights = optim.x
        print "-", optim.fun

    train_preds = ktools.ensemble_preds(all_train_preds, init_weights)
    score = tools.get_mae_loss(targets, train_preds)
    print 'score ens manual:', score, init_weights

    train_preds = ktools.ensemble_preds(all_train_preds, optimised_weights)
    score = tools.get_mae_loss(targets, train_preds)
    print 'score ens with optim:', score, optimised_weights

    print 'generating predictions for the test set'
    all_test_preds = convert_preds_to_list(new_test)
    sub_preds = ktools.ensemble_preds(all_test_preds, optimised_weights)

    sub_file_name = 'stkOptim_x{}_f{}'.format(n_models, n_folds)
    data.generate_simple_kaggle_file(sub_preds, sub_file_name)
    msg = 'score ens:{}, w:{}.weights, file:{}'.format(score, optimised_weights, sub_file_name)
    logger.debug(msg)

