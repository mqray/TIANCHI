# -*- coding: utf-8 -*-
# @Time    : 2019/3/27 13:28
# @Author  : mqray
# @Site    : 
# @File    : q1.py
# @Software: PyCharm


from folium import plugins
import folium
import pandas as pd
import numpy as np
from folium import plugins
import folium
import os
new_data = pd.read_csv(r'E:\TIANCHI\Tips\1carInfo\AA00001.csv')
new_data[(new_data['acc_state']==0)&(new_data['gps_speed']==0)].shape[0]
stoped_state = new_data[(new_data['acc_state']==0)&(new_data['gps_speed']==0)]
stoped_state.head()
stoped_state.drop_duplicates(inplace=True)
stoped_state.shape[0]
stop = new_data.loc[[0]].append(stoped_state)
marks = stop.index
lens = stop.shape[0]
new = pd.DataFrame(marks.to_list())
news = marks.to_list()
l = []
for i in range(len(news)-1):
    dif = news[i+1]-news[i]
#     print(dif)
    if dif<900:
        l.append(news[i])
print(l)

mark1 = list(set(news).difference(set(l)))
np.sort(mark1)
mark2 = [x for x in news if x not in l]
lat_lng = []
for i in range(len(mark2)-1):
    start = mark2[i]
    end = mark2[i+1]
    range_data = new_data[start:end]
    lng = range_data.iloc[:,3:4]#经度
    lat = range_data.iloc[:,4:5]#维度
    # 绘图的时候要将维度放在前面
    lat_lng = lat.join(lng)
    lat_lng_f = lat_lng.values.tolist()
#     lat_lng_format = [list(map(eval,x)) for x in lat_lng_f]
    locations = lat_lng_f
    m = folium.Map(lat_lng_f[0],zoom_start=10)
    route = folium.PolyLine(
        locations,
        weight=3,
        color='red',
        opacity=0.8
    ).add_to(m)
    filename = 'Heatmap'+str(i)+'.html'
    m.save(os.path.join(r'E:\TIANCHI\Tips\explore\2', filename))