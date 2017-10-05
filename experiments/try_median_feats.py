import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from million import data, tools, features
from million import model_params

cache_dir = tools.cache_dir()
seed = 1
evaluate_cv = False
nrounds = 2930


if __name__ == '__main__':
    np.random.seed(seed)
    df = data.load_data(from_cache=True)
    targets = df['logerror'].values

    if evaluate_cv:
        df, targets, train_ixs, test_ixs = data.get_cv_ixs(df, targets)
    else:
        train_ixs, test_ixs = data.get_lb_ixs(targets)

    df = features.add_features(df, train_ixs)
    df = data.select_features(df)
    print df.columns

    df_train, train_targets = df.iloc[train_ixs], targets[train_ixs]
    if evaluate_cv:
        df_test, test_targets = df.iloc[test_ixs], targets[test_ixs]
        eval_set = [(df_test.values, test_targets)]
    else:
        df_test = df.iloc[test_ixs]
        eval_set = [(df_train.values, train_targets)]

    params = model_params.get_ltune7k(num_rounds=nrounds)
    model = LGBMRegressor(**params)
    model.fit(
            df_train.values, train_targets,
            eval_set=eval_set,
            early_stopping_rounds=80
            )
    predictions = model.predict(df_test)
    if evaluate_cv:
        print(tools.get_mae_loss(test_targets, predictions))

    if not evaluate_cv:
        predictions = model.predict(df_test)
        data.generate_simple_kaggle_file(predictions, 'sub_singlelgb_quasies')
