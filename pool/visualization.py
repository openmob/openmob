print('this is a test...')
print('aaaa')


import folium
import pandas as pd
import glob
# from os.path import abspath, dirname
# os.chdir(dirname(abspath(__file__)))
# print(os.getcwd())



if __name__ == '__main__':

    # sample_ = container[(container.pa_name == '浜松ＳＡ（下り）') | (container.pa_name == '浜松ＳＡ（上り）')]
    # sample_ = container[container.pa_name == '市川パーキングエリア']
    sample_ = container[container.pa_name == '平和島パーキングエリア']
    sample_ = sample_.reset_index(drop=True)
    mymap = folium.Map(location=[sample_.lat.mean(), sample_.lon.mean()], zoom_start=15, width=1000, height=1000)
    # folium.PolyLine(df[['Latitude','Longitude']].values, color="red", weight=2.5, opacity=1).add_to(mymap)
    # for coord in sample_[:100][['LATITUDE','LONGITUDE']].values:
    #     folium.CircleMarker(location=[coord[0],coord[1]], radius=2,color='red').add_to(mymap)
    # id_set = sample_.M2M_MODEL_CD.unique()
    color_map = ['purple', 'blue', 'green', 'red', 'orange', 'black', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray',
                 'darkred', 'lightgray']
    type_car = ['none', '大型トラック', '大型バス', '中型トラック', 'none', '小型トラック']
    # for i, id_ in enumerate(sample_):
    #     tmp = sample_[sample_.M2M_MODEL_CD == id_]
    #     my_PolyLine=folium.PolyLine(locations=tmp[['LATITUDE','LONGITUDE']].values,weight=2, color=color_map[i])
    #     mymap.add_child(my_PolyLine)
    for i, coord in enumerate(sample_[['lat', 'lon']].values):
        folium.CircleMarker(location=[coord[0], coord[1]], radius=2, color=color_map[sample_.KBN[i]],
                            popup=
                            "Arrival: " + str(sample_.arrival_time[i]) + '<br>' +
                            "Departure: " + str(sample_.departure_time[i]) + '<br>' +
                            "ID: " + str(sample_.M2M_MODEL_CD[i]) + '<br>'
                                                                    "Type: " + str(type_car[sample_.KBN[i]]) + '<br>', ).add_to(mymap)
    mymap  # shows map inline in Jupyter but takes up full width
    # mymap.save('../visualization/hamamatsu.html')  # saves to html file for display below