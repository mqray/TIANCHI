# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 8:46
# @Author  : mqray
# @Site    : 
# @File    : xgb.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2019/3/18 8:30
# @Author  : mqray
# @Site    :
# @File    : try.py
# @Software: PyCharm

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import datetime
from sklearn.metrics import  mean_absolute_error
from xgboost import XGBRegressor
columns = ['id','auth','age','college','black','4G','use_age','lasted_pay',
           'lasted_pay_money','ave_6','bill_now','surplus','arrears','sensity',
           'chat_num','market','show_3','wanda','sam','movie','tour','gym',
           'netshopping','express','finance','vedio','airplane','subway','vistor','score']
data = pd.read_csv(r'E:\TIANCHI\MOBILE\train_dataset.csv', names=columns)
data.drop(index=0, inplace=True)
data.drop(columns='id', axis=1, inplace=True)
data = pd.DataFrame(data[:-1], dtype=np.float)

y = data['score']
y = pd.DataFrame(y,dtype=np.int)

data.drop(columns='score', inplace=True)

data =  RobustScaler(with_scaling=True).fit_transform(data)
X_train,X_val,y_train,y_val = train_test_split(data,y,test_size=0.2,random_state=3)
#0.06259666782417905
model = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=200, silent=False, objective='reg:gamma')
model.fit(X_train,y_train)

res = model.predict(X_val)

mae_pre = mean_absolute_error(y_val,res)
score = 1/(1+mae_pre)
print(score)

# columns1 = ['id','auth','age','college','black','4G','use_age','lasted_pay',
#            'lasted_pay_money','ave_6','bill_now','surplus','arrears','sensity',
#            'chat_num','market','show_3','wanda','sam','movie','tour','gym',
#            'netshopping','express','finance','vedio','airplane','subway','vistor']
# test_data = pd.read_csv('test_dataset.csv',names=columns1)
# test_data.drop(index=0,inplace=True)
#
# id = test_data['id'].copy()
# test_data.drop(['id'],axis=1,inplace=True)
# test_data = pd.DataFrame(test_data,dtype=np.float)
# # test_data = RobustScaler(with_scaling=True).fit_transform(test_data)
# result = model.predict(test_data)
#
# final = pd.DataFrame(result,columns=['score'])
# id = id.copy().reset_index(drop=True)
# file = pd.concat([id,final],axis=1)
# file.to_csv('xgb_res.csv',index=0,float_format='%.0f',encoding='utf-8')
