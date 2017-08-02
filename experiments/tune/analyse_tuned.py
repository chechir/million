import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from million.experiments.tune.tune_cat import TUNING_RESULTS_PATH
from million import tools

xgb_columns = ['max_depth', 'min_child_weight', 'lambda', 'alpha', 'subsample', 'colsample_bytree', 'eta']
lgb_columns = [
            'learning_rate', 'min_data', 'min_hessian', 'lambda_l1', 'lambda_l2',
            'num_leaves', 'subsample_for_bin', 'min_child_samples', 'max_depth',
            'min_child_weight', 'subsample_freq', 'subsample', 'colsample_bytree'
            ]
cat_columns = ['learning_rate', 'depth', 'l2_leaf_reg', 'rsm', 'bagging_temperature',
            'fold_permutation_block_size', 'gradient_iterations', 'has_time'
            ]

#mlp_columns = ['num_layers', 'dropout', 'dropout_decrease', 'num_units', 'batch_size']
# ss tuner

binent = 'mae'
kelly = 'mse'

def plot_kelly_bin(df):
    pal = sns.color_palette()
    plt.scatter(df[binent], df[kelly], alpha=0.8, color=pal[0], label='Kelly / bin_ent')
    plt.xlabel('bin_ent')
    plt.ylabel('kelly_edge')
    plt.legend()
    plt.show()

def plot_losses_against_params(df, loss):
    pal = sns.color_palette()
    params = cat_columns
    for i, param in enumerate(params):
        if i > 5:
            i = 5
        plt.scatter(df[param], df[loss], color=pal[i], alpha=0.6, label=param)
        plt.xlabel(param)
        plt.ylabel(loss)
        #plt.legend()
        plt.show()

if __name__ == '__main__':
    #df = ss.DDF.from_csv(TUNING_RESULTS_PATH)
    df = tools.read_special_json(TUNING_RESULTS_PATH)
    print 'df shape', df.shape
    best_bin_ix = np.argmin(df[binent])
    best_kelly_ix = np.argmax(df[kelly])
    print df.rowslice(best_bin_ix)
    #print df.rowslice(best_kelly_ix)
    print 'both the same best?:', best_kelly_ix == best_bin_ix
    #plot_kelly_bin(df)
    #loss = binent
    #plot_losses_against_params(df, loss)


