import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor

from million import data, features, tools
from million._config import NULL_VALUE, test_columns, test_dates
from million import model_params

seed = 147
xgb_weight = 0.75


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

    ################
    ##  XGBoost   ##
    ################

    df_train, targets, df_test = data.split_data(df, logerror)

    dtrain = xgb.DMatrix(df_train.values, targets)
    dtest = xgb.DMatrix(df_test.values)

    sub_model = xgb.train(
            model_params.get_xtune11k(),
            dtrain, num_boost_round=105,
            )
    xgb_preds = sub_model.predict(dtest)
    print( "\n XGB predictions:" )
    print( pd.DataFrame(xgb_preds).head() )

    ################
    ##  LightGBM  ##
    ################
    sub_model = LGBMRegressor(**model_params.get_ltune7k())
    sub_model.fit(df_train, targets)
    lgb_preds = sub_model.predict(df_test)
    print( "\n LGB predictions:" )
    print( pd.DataFrame(lgb_preds).head() )

    weights = (xgb_weight, 1-xgb_weight)
    final_preds = tools.ensemble_preds([xgb_preds, lgb_preds], weights)
    data.generate_simple_kaggle_file(final_preds, 'ensemble')

    print( "\n 'Ensemble predictions:" )
    print( pd.DataFrame(final_preds).head() )

