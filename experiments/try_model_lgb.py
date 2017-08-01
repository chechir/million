import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor

from million import data, features, tools
from million._config import NULL_VALUE, test_columns, test_dates
from million import model_params

seed = 147

if __name__ == '__main__':
    np.random.seed(seed)
    df_train, df_test = data.load_data(cache=True)
    df = data.create_fulldf(df_train, df_test)

    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)
    #df = features.add_features(df)

    logerror = df['logerror'].values
    targets = logerror
    df = data.select_features(df)

    print df.columns
    df_full_train, targets, df_test = data.split_data(df, logerror)
    df_train, df_test, train_targets, test_targets = data.split_cv(df_full_train, targets, 0.8)

    dtrain = xgb.DMatrix(df_train.values, train_targets)
    dtest = xgb.DMatrix(df_test.values, test_targets)

    params = model_params.get_ltune7k()
    model = LGBMRegressor(**params)
    model.fit(
            df_train, train_targets,
            eval_set=[(df_test, test_targets)],
            early_stopping_rounds=25
            )
    predictions = model.predict(df_test)
    print(tools.get_mae_loss(test_targets, predictions))

    ###training full:
    df_train, targets, df_test = data.split_data(df, logerror)

    dtrain = xgb.DMatrix(df_train.values, targets)
    dtest = xgb.DMatrix(df_test.values)

    sub_model = LGBMRegressor(**params)
    sub_model.fit(
            df_train, targets,
            )
    predictions = sub_model.predict(df_test)
    data.generate_simple_kaggle_file(predictions, 'sub/sub_singlelgb')

