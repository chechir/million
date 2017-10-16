
import pandas as pd

weight = 0.40

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
    file1 = 'Only_CatBoost.csv' # 0.0641
    file2 = 'sub_20171011_151323_stk_3lvl2_models_x14_f5_wiggle1.1.csv'  # https://www.kaggle.com/aharless/xgboost-using-4th-quarter-for-validation/output (v67)
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = get_ensemble_from_subs(df1, df2)
    print( "\nWriting results to disk ..." )
    df.to_csv('ens_{}_{}.csv'.format(file1[:20], file2), index=False)
