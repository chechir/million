import numpy as np
from catboost import CatBoostRegressor

from million import data, features
from million._config import NULL_VALUE, test_columns, test_dates

seed = 147

def get_catboost():
    model = CatBoostRegressor(learning_rate=1, depth=32, loss_function='RMSE')
    return model

def filter_feats(df):
    not_wanted_feats = ['transactiondate_month', 'transactiondate']
    final_cols = [col for col in df.columns if col not in not_wanted_feats]
    return df[final_cols]

if __name__ == '__main__':
    np.random.seed(seed)
    df_train, df_test = data.load_data(cache=True)
    df = data.create_fulldf(df_train, df_test)
    df = df.fillna(NULL_VALUE)
    df = data.clean_data(df)
    df = data.encode_labels(df)
#    df = features.add_features(df)

    targets = df['logerror'].values
    df = data.select_features(df)
    df = filter_feats(df)
    print df.columns

    df_train, targets, df_test = data.split_data(df, targets)

    model = get_catboost()
    fit_model = model.fit(df_train.values, targets)


