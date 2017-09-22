import numpy as np
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from million import data, tools, features
from million import model_params

cache_dir = tools.cache_dir()
seed = 1
if __name__ == '__main__':
    np.random.seed(seed)
    df = data.load_data(from_cache=True)
    train_ixs, test_ixs = data.get_ixs(df)
    df = features.add_features(df, train_ixs)

    targets = df['logerror'].values
    df = data.select_features(df)
