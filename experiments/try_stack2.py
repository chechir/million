import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# seed = 14
seed = 20
np.random.seed(seed) #keras?

from million import data, tools
from million._config import NULL_VALUE
from million import model_params
import seamless as ss

n_folds = 5
features_2nd = [
        'svrb_preds', 'catb_preds', 'kerasb_preds'
        ]
n_models = len(features_2nd)

cache_dir = tools.cache_dir()


if __name__ == '__main__':
    np.random.seed(seed)
    df = data.load_data(from_cache=True)
    df = tools.remove_ouliers(df)
    targets = df['logerror'].values

    train_ixs, test_ixs = data.get_lb_ixs(targets)
    # df = features.add_features(df, train_ixs)

    df = data.select_features(df)
    print df.columns
    df_train, train_targets = df.iloc[train_ixs], targets[train_ixs]
    df_test = df.iloc[test_ixs]

    feats_small = ['taxamount', 'heatingorsystemtypeid', 'yearbuilt', 'bedroomcnt',
                   'fullbathcnt', 'calculatedbathnbr', 'bathroomcnt',
                   'calculatedfinishedsquarefeet', 'finishedsquarefeet12']

    df_test_small = df_test[feats_small]

    folds = [np.random.randint(0, n_folds) for x in range(len(df_train))]
    preds_train = np.zeros(shape=(len(df_train), n_models))
    preds_test = np.zeros(shape=(len(df_test), n_models))

    train = df_train.values
    for i in range(n_folds):
        train_ix = (np.repeat(i, len(folds)) != folds)
        val_ix = ~ train_ix
        x_train, y_train = train[train_ix, :], train_targets[train_ix]
        x_val, y_val = train[val_ix, :], train_targets[val_ix]

        x_train_small = df_train[feats_small].values[train_ix, :]
        x_val_small = df_train[feats_small].values[val_ix, :]
        assert len(x_train_small) == len(x_train)

        # keras
        keras_ix = 2
        batch_size, epochs = 256, 15
        model = model_params.get_keras(x_train.shape[1])
        history = model.fit(
                x_train, y_train,
                nb_epoch=epochs, batch_size=batch_size,
                validation_data=(x_val, y_val), verbose=2)
        model.history = history
        preds_train[val_ix, keras_ix] = model.predict(x_val).squeeze()
        preds_test[:, keras_ix] += model.predict(df_test.values).squeeze()
        score = tools.get_mae_loss(y_val, preds_train[val_ix, keras_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        # svr !
        svr_ix = 0
        model = SVR(cache_size=600, C=0.1)
        print x_train_small.shape
        model.fit(x_train_small[::10, :], y_train[::10])
        preds_train[val_ix, svr_ix] = model.predict(x_val_small)
        preds_test[:, svr_ix] += model.predict(df_test_small.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, svr_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train),
              len(x_val), i, score))

        # Catboost
        cat_ix = 1
        params = model_params.get_ctune729c()
        params.pop('use_best_model')
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)
        preds_train[val_ix, cat_ix] = model.predict(x_val)
        preds_test[:, cat_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, cat_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train),
              len(x_val), i, score))

    preds_test = preds_test / float(n_folds)
    new_train = df_train.copy()
    new_test = df_test.copy()

    model_ixs = [svr_ix, cat_ix, keras_ix]
    for model_pred in zip(features_2nd, model_ixs):
        new_train[model_pred[0]] = preds_train[:, model_pred[1]]
        new_test[model_pred[0]] = preds_test[:, model_pred[1]]

    print('saving first level preds')
    ss.io.write_pickle(new_train[features_2nd], cache_dir + 'ps2_train2_2ndx{}_f{}.pkl'.format(n_models, n_folds))
    ss.io.write_pickle(new_test[features_2nd], cache_dir + 'ps2_test2_2ndx{}_f{}.pkl'.format(n_models, n_folds))

    # second level predictions
    model = LinearRegression(n_jobs=-1)
    model.fit(new_train[features_2nd], train_targets)
    print(tools.get_mae_loss(train_targets, model.predict(new_train[features_2nd])))
    print(tools.get_mae_loss(train_targets, new_train['catb_preds'].values))

    sub_preds = model.predict(new_test[features_2nd])
    sub_preds = np.clip(sub_preds, -0.5, 0.5)
    print sub_preds[0:20]
    data.generate_simple_kaggle_file(sub_preds, 'stack_new_{}'.format(n_folds))

