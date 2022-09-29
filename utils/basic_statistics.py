import glob

import pandas as pd
import tqdm
from matplotlib import pyplot as plt, dates as mdates

tqdm.tqdm.pandas()

if __name__ == '__main__':
    file_list = glob.glob('../data/*.csv')

    statistics = pd.DataFrame()
    for i, file in enumerate(file_list):
        tmp = pd.read_csv(file)
        tmp = tmp[tmp.lon != 'lon']
        tmp['arrival_time'] = pd.to_datetime(tmp.arrival_time)
        statistics.loc[i, 'date'] = tmp.loc[i, 'arrival_time'].date()
        statistics.loc[i, 'records'] = len(tmp)
        statistics.loc[i, 'ID_num'] = len(tmp.M2M_MODEL_CD.unique())

    statistics = statistics.sort_values(by='date')

    fig = plt.figure(dpi=300, figsize=(15, 5))
    plt.plot(statistics.date, statistics.records, label='Total records', c='g', marker='o')
    plt.plot(statistics.date, statistics.ID_num, label='ID number', c='b', marker='x')
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    # locator = mdates.DayLocator()
    # ax.xaxis.set_major_locator(locator)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.savefig('./statistics.png')
