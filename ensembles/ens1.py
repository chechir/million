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
    file1 = 'sub20170718_152426.csv' # 0.0644151
    file2 = 'sub20170718_163420_kaggle.csv' # 0.0644121 (maybe)

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = get_ensemble_from_subs(df1, df2)

    print( "\nWriting results to disk ..." )
    df.to_csv('ens_{}_{}.csv'.format(file1, file2), index=False)


