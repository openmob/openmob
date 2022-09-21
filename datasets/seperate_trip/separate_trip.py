import pandas as pd


def separate_trip(file='../dataset_tsmc2014/dataset_TSMC2014_TKY.txt', length=10):
    data = pd.read_table(file, encoding='latin-1', header=None)
    data.columns = ['uid', 'vid', 'vcid', 'vname', 'lat', 'lon', 'offset', 'utctime']
    for uid_ in data.uid.unique()[:length]:
        data[data.uid == uid_].to_csv('./{}.csv'.format(uid_), index=False)
    return


if __name__ == '__main__':
    separate_trip()
