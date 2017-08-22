import numpy as np
from sklearn.metrics import mean_squared_error

from million import data, features, tools
from million._config import NULL_VALUE, test_columns, test_dates
from million import model_params

cv_flag = True
seed = 1
cv_split_ratio = 0.8
n_bags = 1

LOG_FILE = tools.experiments() + 'millions_try_xgb_.txt'
logger = tools.get_logger(LOG_FILE)

epochs = 30
batch_size = 64

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
    df = df.drop(['assessmentyear'], axis=1)

    print df.columns
    if cv_flag:
        df_full_train, targets, df_test = data.split_data(df, logerror)
        df_train, df_val, train_targets, val_targets = data.split_cv(df_full_train, targets, cv_split_ratio)

        cv_preds = np.repeat(0., len(df_val))
        for i in range(n_bags):
            x_train, x_val = tools.normalise_data(df_train.values, df_val.values)
            model = model_params.get_keras(x_train.shape[1])
            history = model.fit(
                    x_train, train_targets,
                    nb_epoch=epochs, batch_size=batch_size,
                            validation_data=(x_val, val_targets), verbose=2)
            model.history = history
            cv_preds += model.predict(x_val).squeeze()
        cv_preds /= float(n_bags)

        mae = tools.get_mae_loss(val_targets, cv_preds)
        mse = mean_squared_error(val_targets, cv_preds)
        msg = 'mae: {}, mse: {}, keras! train_data ratio: {}, bags:{}, epochs:{}'.format(mae, mse, cv_split_ratio, n_bags, epochs)
        print(msg), logger.debug(msg)

    else:
        print 'hola'
        ###training full:
        #data.generate_simple_kaggle_file(final_preds, 'bagged_{}'.format(n_bags))

