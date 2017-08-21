
import pandas as pd

from million import tools

weight = 0.8

subs_dir = tools.dropbox() + 'million/sub/'

def get_ensemble_from_subs(df1, df2, weight):
    df1.sort_values('ParcelId', axis=0, inplace=True)
    df2.sort_values('ParcelId', axis=0, inplace=True)
    cols = df1.columns[1:len(df1.columns)]
    df = df1.copy()
    for col in cols:
        df[col] = df1[col]*weight + df2[col]*(1 - weight)
    return df

if __name__ == '__main__':
    print( "\nReading data from disk ...")
    file1 = 'sub_kaggle_plus_bagged_cat20_.7120170803_075809.csv' # 0.0643453
    file2 = 'sub_20170821_094102_stacked_components_5.csv' # 0.0645256

    df1 = pd.read_csv(subs_dir + file1)
    df2 = pd.read_csv(subs_dir + file2)
    df = get_ensemble_from_subs(df1, df2, weight)

    print( "\nWriting results to disk ..." )
    df.to_csv(subs_dir + 'ens_{}_{}_{}.csv'.format(file1[:50], file2[:50], weight), index=False)

