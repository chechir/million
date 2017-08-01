import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from million import data, features, tools
from million._config import NULL_VALUE, test_columns, test_dates
from million import model_params

cv_flag = False
BASELINE_PRED = 0.0110   # Baseline based on mean of training data, per Oleg
BASELINE_WEIGHT = 0.0059

seed = 147
cv_split_ratio = 0.8
n_bags = 10

def delete_some_outliers(df, targets):
    outlier_ixs = (targets > 0.419) | (targets < -0.4)
    filter_ixs = np.array([(np.random.normal() > 0.5) & o for o in outlier_ixs])
    return df.iloc[~filter_ixs], targets[~filter_ixs]

def drop_some_columns(df, test_df):
    cols_to_drop = ['fireplacecnt']
    result_df = df.copy()
    result_test_df = test_df.copy()
    if np.random.normal() > 0.5:
        result_df = df.drop(cols_to_drop, axis=1)
        result_test_df = test_df.drop(cols_to_drop, axis=1)
    return result_df, result_test_df

LOG_FILE = tools.experiments() + 'millions_try_xgb_.txt'
if __name__ == '__main__':
    np.random.seed(seed)
    logger = tools.get_logger(LOG_FILE)
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
    if cv_flag:
        df_full_train, targets, df_test = data.split_data(df, logerror)
        df_train, df_test, train_targets, test_targets = data.split_cv(df_full_train, targets, cv_split_ratio)

        dtest = xgb.DMatrix(df_test.values, test_targets)
        dtrain = xgb.DMatrix(df_train.values, train_targets)

        params = model_params.get_xtune11k()
        cv_preds = np.repeat(0, len(df_test))
        num_boost_rounds = 110
        for i in range(n_bags):
            watchlist  = [(dtrain,'train'),(dtest,'eval')]
            model = xgb.train(
                    params,
                    dtrain, num_boost_round=num_boost_rounds,
                    evals=watchlist,
                    early_stopping_rounds=50
                    )
            cv_preds = model.predict(dtest) + cv_preds

            #prepare for the next iteration
            df_bag, bag_targets = delete_some_outliers(df_train, train_targets)
            dtrain = xgb.DMatrix(df_bag.values, bag_targets)
            print(i, df_bag.shape)
            #params['seed'] = i
            num_boost_rounds = 155
        cv_preds = cv_preds / n_bags

        mae = tools.get_mae_loss(test_targets, cv_preds)
        mse = mean_squared_error(test_targets, cv_preds)
        msg = 'mae: {}, mse: {}, train_data ratio: {}, bags:{}, r:{}'.format(mae, mse, cv_split_ratio, n_bags, num_boost_rounds)
        print(msg), logger.debug(msg)

    else:
        ###training full:
        df_train, targets, df_test = data.split_data(df, logerror)

        dtest = xgb.DMatrix(df_test.values)
        dtrain = xgb.DMatrix(df_train.values, targets)
        params = model_params.get_xtune11k()

        sub_preds = np.repeat(0, len(df_test))
        num_boost_rounds = 110
        for i in range(n_bags):
            model = xgb.train(
                    params,
                    dtrain, num_boost_round=num_boost_rounds,
                    )
            sub_preds = model.predict(dtest) + sub_preds

            #prepare for the next iteration
            df_bag, bag_targets = delete_some_outliers(df_train, targets)
            dtrain = xgb.DMatrix(df_bag.values, bag_targets)
            print(i, df_bag.shape)
            num_boost_rounds = 155
            #params['seed'] = i
        sub_preds = sub_preds / n_bags

        print sub_preds[0:10]
        weights = (1-BASELINE_WEIGHT, BASELINE_WEIGHT)
        print weights
        final_preds = tools.ensemble_preds([sub_preds, np.repeat(BASELINE_PRED, len(sub_preds))], weights)
        print final_preds[0:10]

        data.generate_simple_kaggle_file(final_preds, 'bagged_{}'.format(n_bags))

