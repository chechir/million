
y_mean = 0.0102590

def get_xtune11k(y_mean=y_mean):
    xgb_params = {
        'objective': 'reg:linear',
        'eta': 0.015,
        'max_depth': 12,
        'subsample': 0.454482969106,
        'colsample_bytree': 0.414702544498,
        'eval_metric': 'mae',
        'lambda': 3.48507025002,
        'alpha': 3.74462055194,
        'seed':213,
        'min_child_weight': 81,
        'base_score': 0.0102590376627,
        'silent': 1
    }
    return xgb_params

def get_ltune7k(y_mean=y_mean):
    lgb_params = {
        'n_estimators': 1830,
        'boosting_type': 'gbdt',
        'metric': 'mae',
        'objective': 'regression',
        'num_threads':4,
        'max_depth': 10,
        'subsample_for_bin': 2191288.0,
        'subsample_freq': 9,
        'subsample': 0.385551103749,
        'lambda_l1': 2.66397707793,
        'lambda_l2': 3.91942260463,
        'num_leaves': 128,
        'min_child_samples': 47.362342627,
        'min_child_weight': 115,
        'min_split_gain': 0,
        'learning_rate': 0.00199957544464,
        'min_data':178,
        'colsample_bytree': 0.61111511025,
        'min_hessian': 0.145983080031,
        'seed': 3073881,
        }
    return lgb_params

def get_ctune80(y_mean=y_mean):
    cat_params = {
        'iterations': 8352 + 10,
        'thread_count': 8,
        'loss_function': 'MAE',
        'auto_stop_pval': 0.0001,
        'learning_rate': 0.00321990477898,
        'depth': 5,
        'l2_leaf_reg': 3,
        'rsm': 0.407419112405,
        'bagging_temperature': 0.847795225304,
        'fold_permutation_block_size': 2,
        'gradient_iterations': 1,
        'random_seed': 952,
        'has_time': False,
        'use_best_model': True,
        'verbose': True,
        #'ctr_border_count': 5050,
        #'max_ctr_complexity': 4,
        'name': 'experiment',
    }
    return cat_params

def get_ctune293(y_mean=y_mean):
    cat_params = {
        'iterations': 7760,
        'thread_count': 8,
        'loss_function': 'MAE',
        'auto_stop_pval': 0.0001,
        'learning_rate': 0.0027782225151,
        'depth': 6,
        'l2_leaf_reg': 5,
        'rsm': 0.589884179754,
        'bagging_temperature': 0.660536010706,
        'fold_permutation_block_size': 2,
        'gradient_iterations': 1,
        'random_seed': 18,
        'has_time': False,
        'use_best_model': True,
        'verbose': True,
        #'ctr_border_count': 5050,
        #'max_ctr_complexity': 4,
        'name': 'experiment',
    }
    return cat_params

