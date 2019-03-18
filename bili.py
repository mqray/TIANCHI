# -*- coding: utf-8 -*-
# @Time    : 2019/3/17 13:46
# @Author  : mqray
# @Site    : 
# @File    : bili.py
# @Software: PyCharm

from sklearn.preprocessing import  MinMaxScaler
import pandas as pd
data = [[-1,2],[-0.5,6],[0,10],[1,18]]
pd.DataFrame(data)

scaler = MinMaxScaler()#实例化
scaler = scaler.fit(data)#fit，生成min和max
res = scaler.transform(data)#通过接口导出结果
print(res)
#
# res_ = scaler.fit_transform((data))#训练和导出结果一步实现
# scaler.inverse_transform(res)#归一化后结果逆转
#
# data = [[-1,2],[-0.5,6],[0,10],[1,18]]
# scaler = MinMaxScaler(feature_range=[5,10])#实例化，但是将数据归一化到5-10中