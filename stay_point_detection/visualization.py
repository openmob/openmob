import folium


def visualization_stay_points(raw_data, sp, number):
    map_ = folium.Map(location=[35.65, 139.7], zoom_start=12)

    uids = sp.user_id.unique()[:number]

    for uid in uids:
        tmp = sp[sp.user_id == uid]
        tmp = tmp.reset_index(drop=True)
        folium.PolyLine(zip(raw_data[raw_data.user_id == uid].lat.values, raw_data[raw_data.user_id == uid].lon.values), color="gray", weight=1., opacity=1).add_to(map_)
        for i, coord in enumerate(tmp[['lat', 'lon']].values):
            popup_content = 'user_id: {}<br>venue name: {}<br>arrival time: {}<br>departure time:{}'.format(
                int(tmp.loc[i, 'user_id']),
                tmp.loc[i, 'venue_name'],
                tmp.loc[i, 'arrival_time'],
                tmp.loc[i, 'departure_time'])
            iframe = folium.IFrame(popup_content)
            popup1 = folium.Popup(iframe,
                                  min_width=280,
                                  max_width=120)
            folium.CircleMarker(location=[coord[0], coord[1]], radius=10, color='red',
                                popup=popup1).add_to(map_)
    return map_


if __name__ == '__main__':
    m = visualization_stay_points('../functions/stay_point_detection/stay_points/', number=3)
    m.save('./stay_points.html')
    print('here')
