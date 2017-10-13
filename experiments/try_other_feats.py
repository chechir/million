import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from million import data, tools, features
from million import model_params

cache_dir = tools.cache_dir()
seed = 1
evaluate_cv = True
nrounds = 2930

# bestTest = 0.06860590443
# bestIteration = 3130
# bad model
# basic cols (much better): 0.068622230
# with quasi median city  : 0.0687510447173

if __name__ == '__main__':
    np.random.seed(seed)
    df = data.load_data(from_cache=True)
    targets = df['logerror'].values

    if evaluate_cv:
        df, targets, train_ixs, test_ixs = data.get_cv_ixs(df, targets)
    else:
        train_ixs, test_ixs = data.get_lb_ixs(targets)

    df = features.add_features_exp(df, train_ixs)
    df = data.select_features(df)
    print df.columns

    df_train, train_targets = df.iloc[train_ixs], targets[train_ixs]
    if evaluate_cv:
        df_test, test_targets = df.iloc[test_ixs], targets[test_ixs]
        eval_set = [df_test.values, test_targets]
    else:
        df_test = df.iloc[test_ixs]
        eval_set = [df_train.values, train_targets]

    params = model_params.get_ctune293x()
    params.pop('use_best_model')
    model = CatBoostRegressor(**params)
    model.fit(df_train.values, train_targets,
              eval_set=eval_set,
              )
    predictions = model.predict(df_test)
    if evaluate_cv:
        print(tools.get_mae_loss(test_targets, predictions))

    if not evaluate_cv:
        predictions = model.predict(df_test)
#        predictions = predict_by_period(df_test, model)
        data.generate_simple_kaggle_file(predictions, 'sub_singlelgb_quasies')
