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
        'cat_preds', 'xgb_preds', 'lgb_preds', 'ker_preds', 'cat2_preds',
        'et_preds', 'knn_preds', 'ridge_preds', 'rfr_preds', 'svr_preds',
        'knn2_preds'
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

        # svr !
        svr_ix = 9
        model = SVR()
        print x_train_small.shape
        model.fit(x_train_small[-4333:, :], y_train[-4333:])
        preds_train[val_ix, svr_ix] = model.predict(x_val_small)
        preds_test[:, svr_ix] += model.predict(df_test_small.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, svr_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        # Catboost
        cat_ix = 0
        params = model_params.get_ctune293()
        params.pop('use_best_model')
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)
        preds_train[val_ix, cat_ix] = model.predict(x_val)
        preds_test[:, cat_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, cat_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train),
              len(x_val), i, score))

        # xgboost
        xgb_ix = 1
        params = model_params.get_xtune11k()
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(df_test.values)
        model = xgb.train(
                model_params.get_xtune11k(),
                dtrain, num_boost_round=120,
                )
        preds_train[val_ix, xgb_ix] = model.predict(xgb.DMatrix(x_val))
        preds_test[:, xgb_ix] += model.predict(dtest)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, xgb_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train),
              len(x_val), i, score))

        # lgb
        lgb_ix = 2
        params = model_params.get_ltune7k()
        model = LGBMRegressor(**model_params.get_ltune7k())
        model.fit(x_train, y_train)
        preds_train[val_ix, lgb_ix] = model.predict(x_val)
        preds_test[:, lgb_ix] += model.predict(df_test_small.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, lgb_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        # keras
        keras_ix = 3
        batch_size, epochs = 64, 30
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

        # Catboost 2!
        cat2_ix = 4
        params = model_params.get_ctune114b()
        params.pop('use_best_model')
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)
        preds_train[val_ix, cat2_ix] = model.predict(x_val)
        preds_test[:, cat2_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, cat2_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        # Extra trees !
        et_ix = 5
        model = ExtraTreesRegressor(n_estimators=120, max_depth=5)
        model.fit(x_train, y_train)
        preds_train[val_ix, et_ix] = model.predict(x_val)
        preds_test[:, et_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, et_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train),
              len(x_val), i, score))

        # knn !
        knn_ix = 6
        model = KNeighborsRegressor(n_neighbors=20)
        model.fit(x_train, y_train)
        preds_train[val_ix, knn_ix] = model.predict(x_val)
        preds_test[:, knn_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, knn_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        # ridge !
        ridge_ix = 7
        model = Ridge()
        model.fit(x_train, y_train)
        preds_train[val_ix, ridge_ix] = model.predict(x_val)
        preds_test[:, ridge_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, ridge_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        # rfr !
        rfr_ix = 8
        model = RandomForestRegressor(n_estimators=100)
        model.fit(x_train, y_train)
        preds_train[val_ix, rfr_ix] = model.predict(x_val)
        preds_test[:, rfr_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, rfr_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        # knn !
        knn2_ix = 10
        model = KNeighborsRegressor(n_neighbors=80)
        model.fit(x_train, y_train)
        preds_train[val_ix, knn2_ix] = model.predict(x_val)
        preds_test[:, knn2_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, knn2_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train),
              len(x_val), i, score))

    preds_test = preds_test / float(n_folds)
    new_train = df_train.copy()
    new_test = df_test.copy()

    model_ixs = [cat_ix, xgb_ix, lgb_ix, keras_ix, cat2_ix, et_ix, knn_ix, ridge_ix, rfr_ix, svr_ix, knn2_ix]
    for model_pred in zip(features_2nd, model_ixs):
        new_train[model_pred[0]] = preds_train[:, model_pred[1]]
        new_test[model_pred[0]] = preds_test[:, model_pred[1]]

    print('saving first level preds')
    ss.io.write_pickle(new_train[features_2nd], cache_dir + 'ps2_train_2ndx{}_f{}.pkl'.format(n_models, n_folds))
    ss.io.write_pickle(new_test[features_2nd], cache_dir + 'ps2_test_2ndx{}_f{}.pkl'.format(n_models, n_folds))

    # second level predictions
    model = LinearRegression(n_jobs=-1)
    model.fit(new_train[features_2nd], train_targets)
    print(tools.get_mae_loss(train_targets, model.predict(new_train[features_2nd])))

    sub_preds = model.predict(new_test[features_2nd])
    sub_preds = np.clip(sub_preds, -0.5, 0.5)
    print sub_preds[0:20]
    data.generate_simple_kaggle_file(sub_preds, 'stack_new_{}'.format(n_folds))

