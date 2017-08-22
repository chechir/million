import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

from million import data, features, tools
from million._config import NULL_VALUE, test_columns, test_dates
from million import model_params

cv_flag = True
BASELINE_PRED = 0.0110   # Baseline based on mean of training data, per Oleg
BASELINE_WEIGHT = 0.0059

seed = 147
cv_split_ratio = 0.8
n_bags = 1

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

    params = model_params.get_ctune163b()
    print df.columns
    if cv_flag:
        df_full_train, targets, df_test = data.split_data(df, logerror)
        df_train, df_test, train_targets, test_targets = data.split_cv(df_full_train, targets, cv_split_ratio)

        cv_preds = np.repeat(0, len(df_test))
        for i in range(n_bags):
            model = CatBoostRegressor(**params)
            eval_set=[df_test.values, test_targets]
            model.fit(df_train.values, train_targets, eval_set=eval_set)

            predictions = model.predict(df_test)
            mae = tools.get_mae_loss(test_targets, predictions)
            mse = mean_squared_error(test_targets, predictions)

            cv_preds = model.predict(df_test.values) + cv_preds

            #prepare for the next iteration
            #df_bag, bag_targets = delete_some_outliers(df_train, train_targets)
            #print(i, df_bag.shape)
            #params['seed'] = i
        cv_preds = cv_preds / n_bags

        mae = tools.get_mae_loss(test_targets, cv_preds)
        mse = mean_squared_error(test_targets, cv_preds)
        msg = 'mae: {}, mse: {}, train_data ratio: {}, bags:{}, r:{}'.format(mae, mse, cv_split_ratio, n_bags, params['iterations'])
        print(msg), logger.debug(msg)

    else:
        ###training full:
        df_train, targets, df_test = data.split_data(df, logerror)
        params.pop('use_best_model')

        sub_preds = np.repeat(0, len(df_test))
        for i in range(n_bags):
            model = CatBoostRegressor(**params)
            model.fit(df_train.values, targets)
            sub_preds = model.predict(df_test.values) + sub_preds
            #prepare for the next iteration
            #df_bag, bag_targets = delete_some_outliers(df_train, targets)
            #params['seed'] = i
        sub_preds = sub_preds / n_bags

        print sub_preds[0:10]
        weights = (1-BASELINE_WEIGHT, BASELINE_WEIGHT)
        print weights
        final_preds = tools.ensemble_preds([sub_preds, np.repeat(BASELINE_PRED, len(sub_preds))], weights)
        print final_preds[0:10]

        data.generate_simple_kaggle_file(final_preds, '293cat_{}'.format(n_bags))

