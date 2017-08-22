import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from million import data, tools
from million._config import NULL_VALUE
from million import model_params

n_folds = 5
features_2nd = ['cat_preds', 'xgb_preds', 'lgb_preds', 'ker_preds', 'cat2_preds']
n_models = len(features_2nd)

seed = 14
cache_dir = tools.cache_dir()

if __name__ == '__main__':
    np.random.seed(seed)
    df_train, df_test = data.load_data(cache=True)
    df = data.create_fulldf(df_train, df_test)

    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)

    logerror = df['logerror'].values
    targets = logerror
    df = data.select_features(df)

    df_train, targets, df_test = data.split_data(df, logerror)

    folds = [np.random.randint(0, n_folds) for x in range(len(df_train))]
    preds_train = np.zeros(shape = (len(df_train), n_models))
    preds_test = np.zeros(shape = (len(df_test), n_models))

    train = df_train.values
    for i in range(n_folds):
        train_ix = (np.repeat(i, len(folds)) != folds)
        val_ix = ~ train_ix
        x_train, y_train = train[train_ix,:], targets[train_ix]
        x_val, y_val = train[val_ix,:], targets[val_ix]

        ######## Catboost
        cat_ix = 0
        params = model_params.get_ctune293()
        params.pop('use_best_model')
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)
        preds_train[val_ix, cat_ix] = model.predict(x_val)
        preds_test[:, cat_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, cat_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        ######## xgboost
        xgb_ix = 1
        params = model_params.get_xtune11k()
        dtrain = xgb.DMatrix(x_train, y_train)
        dtest = xgb.DMatrix(df_test.values)
        model = xgb.train(
                model_params.get_xtune11k(),
                dtrain, num_boost_round = 120,
                )
        preds_train[val_ix, xgb_ix] = model.predict(xgb.DMatrix(x_val))
        preds_test[:, xgb_ix] += model.predict(dtest)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, xgb_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        ######## lgb
        lgb_ix = 2
        params = model_params.get_ltune7k()
        model = LGBMRegressor(**model_params.get_ltune7k())
        model.fit(x_train, y_train)
        preds_train[val_ix, lgb_ix] = model.predict(x_val)
        preds_test[:, lgb_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, lgb_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

        ######## keras
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

        ######## Catboost 2!
        cat2_ix = 4
        params = model_params.get_ctune80()
        params.pop('use_best_model')
        model = CatBoostRegressor(**params)
        model.fit(x_train, y_train)
        preds_train[val_ix, cat_ix] = model.predict(x_val)
        preds_test[:, cat_ix] += model.predict(df_test.values)
        score = tools.get_mae_loss(y_val, preds_train[val_ix, cat_ix])
        print('train rows:{}, val rows:{}, fold:{}, score:{}'.format(len(x_train), len(x_val), i, score))

    preds_test = preds_test / float(n_folds)
    new_train = df_train.copy()
    new_test = df_test.copy()

    model_ixs = [cat_ix, xgb_ix, lgb_ix, keras_ix, cat2_ix]
    for model_pred in zip(features_2nd, model_ixs):
        new_train[model_pred[0]] = preds_train[:, model_pred[1]]
        new_test[model_pred[0]] = preds_test[:, model_pred[1]]

    print('saving first level preds')
    tools.write_pickle(new_train[features_2nd], cache_dir + 'ps_train_2ndx5_f{}.pkl'.format(n_folds))
    tools.write_pickle(new_test[features_2nd], cache_dir + 'ps_test_2ndx5_f{}.pkl'.format(n_folds))

    ## second level predictions
    model = LinearRegression(n_jobs=-1)
    model.fit(new_train[features_2nd], targets); print('fit...')
    print(tools.get_mae_loss(targets, model.predict(new_train[features_2nd])))

    sub_preds = model.predict(new_test[features_2nd])
    sub_preds = np.clip(sub_preds, -0.5, 0.5)
    print sub_preds[0:20]
    data.generate_simple_kaggle_file(sub_preds, 'stacked_{}'.format(n_folds))

