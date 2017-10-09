import pandas as pd

weight = 0.60


def get_ensemble_from_subs(df1, df2):
    df1.sort_values('ParcelId', axis=0, inplace=True)
    df2.sort_values('ParcelId', axis=0, inplace=True)
    cols = df1.columns[1:len(df1.columns)]
    df = df1.copy()
    for col in cols:
        df[col] = df1[col]*weight + df2[col]*(1 - weight)
    return df


if __name__ == '__main__':
    print( "\nReading data from disk ...")
    file1 = 'ens_.5keras_sub_kaggle_plus_bagged_cat20_.7120170803_075809.cs_sub_20170825_085245_stkComp_x7_f5.csv_0.82.csv' # 0.0643207
    file2 = 'sub_20171009_072359_stkOptim_x11_f5.csv'  # https://www.kaggle.com/aharless/xgboost-using-4th-quarter-for-validation/output (v67)
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = get_ensemble_from_subs(df1, df2)
    print( "\nWriting results to disk ..." )
    df.to_csv('ens_{}_{}.csv'.format(file1[:20], file2), index=False)
