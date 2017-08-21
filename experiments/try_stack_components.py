import numpy as np

from million import tools, data
from million._config import NULL_VALUE
from million.experiments.try_stack import n_folds
from kfuncs import tools as ktools

cache_dir = tools.cache_dir()

if __name__ == '__main__':
    df_train, df_test = data.load_data(cache=True)
    df = data.create_fulldf(df_train, df_test)

    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)

    logerror = df['logerror'].values
    targets = logerror
    df = data.select_features(df)

    df_train, targets, df_test = data.split_data(df, logerror)

    new_train = tools.read_pickle(cache_dir + 'ps_train_2nd_folds_{}.pkl'.format(n_folds))
    new_test = tools.read_pickle(cache_dir + 'ps_test_2nd_folds_{}.pkl'.format(n_folds))

    print 'score cat',  tools.get_mae_loss(targets, new_train['cat_preds'])
    print 'score xgb',  tools.get_mae_loss(targets, new_train['xgb_preds'])
    print 'score lgb',  tools.get_mae_loss(targets, new_train['lgb_preds'])

    weights = [.88, .04, .08]
    final_preds = ktools.ensemble_preds(
            (new_train['cat_preds'], new_train['xgb_preds'], new_train['lgb_preds']),
            weights
            )
    print 'score ensemble',  tools.get_mae_loss(targets, final_preds)

    print 'generating predictions for the test set'
    sub_preds = ktools.ensemble_preds(
            (new_test['cat_preds'],new_test['xgb_preds'],new_test['lgb_preds']),
            weights
            )
    data.generate_simple_kaggle_file(sub_preds, 'stacked_components_{}'.format(n_folds))







