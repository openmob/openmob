import folium
import pandas as pd
import glob
import os
import sys
print(os.getcwd())
sys.path.insert(1, '../functions/stay_point_detection/')
from stay_point_detection_process import load_data_tsmc

def visualization_stay_points(stay_points_folder, raw_data_file, number):
    filelist = glob.glob(stay_points_folder + '*.csv')
    print(filelist)

    m = folium.Map(location=[35.65, 139.7], zoom_start=10)
    # tooltip = "Click Here For More Info"
    raw_data = load_data_tsmc(raw_data_file)

    if number > len(filelist):
        print('number of files larger that total stay point files...')
    for file in filelist[:number]:
        data = pd.read_csv(file)
        tmp = raw_data[raw_data.user_id == data.loc[0, 'user_id']]
        folium.PolyLine(zip(tmp.lat.values, tmp.lon.values), color="black", weight=2.5, opacity=1).add_to(m)
        for i, coord in enumerate(data[['lat', 'lon']].values):
            popup_content = 'user_id: {}<br>venue name: {}<br>arrival time: {}<br>departure time:{}'.format(int(data.loc[i, 'user_id']),
                                                                                                                data.loc[i, 'venue_name'],
                                                                                               data.loc[i, 'arrival_time'],
                                                                                               data.loc[i, 'departure_time'])
            iframe = folium.IFrame(popup_content)
            popup1 = folium.Popup(iframe,
                                  min_width=280,
                                  max_width=120)
            folium.CircleMarker(location=[coord[0], coord[1]], radius=10, color='red',
                                popup=popup1).add_to(m)
    return m


if __name__ == '__main__':
    m = visualization_stay_points('../functions/stay_point_detection/stay_points/', raw_data_file='../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt', number=1)
    m.save('./stay_points.html')
    print('here')
