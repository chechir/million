import numpy as np

from million import tools, data
from million._config import NULL_VALUE
from million.experiments.try_stack import n_folds, n_models
from kfuncs import tools as ktools


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

    new_train0 = tools.read_pickle(cache_dir + 'ps_train_2ndx{}_f{}.pkl'.format(5, n_folds))
    new_test0 = tools.read_pickle(cache_dir + 'ps_test_2ndx{}_f{}.pkl'.format(5, n_folds))
    new_train['cat_weird'] = new_train['cat_preds'] + new_train['ker_preds']
    new_test['cat_weird'] = new_test['cat_preds'] + new_test['ker_preds']

    print 'score cat',  tools.get_mae_loss(targets, new_train['cat_preds'])
    print 'score xgb',  tools.get_mae_loss(targets, new_train['xgb_preds'])
    print 'score lgb',  tools.get_mae_loss(targets, new_train['lgb_preds'])
    print 'score keras',  tools.get_mae_loss(targets, new_train['ker_preds'])
    print 'score cat2',  tools.get_mae_loss(targets, new_train['cat2_preds'])
    print 'score cat3',  tools.get_mae_loss(targets, new_train['cat3_preds'])
    print 'score cat4',  tools.get_mae_loss(targets, new_train['cat4_preds'])
    print 'score cat weird',  tools.get_mae_loss(targets, new_train['cat_weird'])

    train0 = np.zeros(len(new_train))
    test0 = np.zeros(len(new_test))

    #weirdness didnt work too well with 0.4, try 0.55
    #weights = [0.34, 0.02, 0.06, 0.08, 0.02, 0.08, 0.42, .02]  (best cv legal)
    #weights = [.55, .02, .07, 0.09, 0.02] (weird submission)
    weights = [.50, .02, .07, 0.18, 0.02, 0.07, 0.06, 0.03, 0.05]
    final_preds = ktools.ensemble_preds(
            (
                new_train['cat_weird'],
                new_train['xgb_preds'],
                new_train['lgb_preds'], new_train['ker_preds'],
                train0,
                new_train['cat_preds'], new_train['cat2_preds'],
                new_train['cat3_preds'], new_train['cat4_preds'],
            ),
            weights
            )
    score = tools.get_mae_loss(targets, final_preds)
    print 'score ens:', score, weights

    print 'generating predictions for the test set'
    sub_preds = ktools.ensemble_preds(
            (
                new_test['cat_weird'],
                new_test['xgb_preds'],
                new_test['lgb_preds'], new_test['ker_preds'],
                test0,
                new_test['cat_preds'], new_test['cat2_preds'],
                new_test['cat3_preds'], new_test['cat4_preds'],
            ),
            weights
            )
    sub_file_name = 'stkComp_x{}_f{}'.format(n_models, n_folds)
    data.generate_simple_kaggle_file(sub_preds, sub_file_name)
    msg = 'score ens:{}, w:{}.weights, file:{}'.format(score, weights, sub_file_name)
    logger.debug(msg)

