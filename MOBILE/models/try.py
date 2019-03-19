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
from sklearn import  linear_model
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import  SVR
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.preprocessing import Binarizer
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error
columns = ['id','auth','age','college','black','4G','use_age','lasted_pay',
           'lasted_pay_money','ave_6','bill_now','surplus','arrears','sensity',
           'chat_num','market','show_3','wanda','sam','movie','tour','gym',
           'netshopping','express','finance','vedio','airplane','subway','vistor','score']
train = pd.read_csv(r'E:\TIANCHI\MOBILE\train_dataset.csv',names=columns)
train.drop(index=0,inplace=True)
train.drop(columns='id',axis=1,inplace=True)

y = train['score']
other = train.drop(columns='score')
# train_copy = pd.DataFrame(train.iloc[:,:],dtype=np.float)
# use_age = train_copy['use_age']
# lasted_pay_money = train_copy['lasted_pay_money']
# lasted_pay = train_copy['lasted_pay']
# ave_6 = train_copy['ave_6']
# bill_now = train_copy['bill_now']
# chat_num = train_copy['chat_num']
# show_3 = train_copy['show_3']
# movie = train_copy['movie']
# tour = train_copy['tour']
# other = pd.concat([use_age,lasted_pay_money,lasted_pay,ave_6,bill_now,chat_num,show_3,movie,tour],axis=1)


X =  RobustScaler(with_scaling=True).fit_transform(other)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=3)

model = RandomForestRegressor()
model.fit(X_train,y_train)
start = datetime.datetime.now()

res = model.predict(X_val)

mae_pre = mean_absolute_error(y_val,res)
score = 1/(1+mae_pre)
print(score)
#
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
# test_data = RobustScaler(with_scaling=True).fit_transform(test_data)
# result = model.predict(test_data)
#
# final = pd.DataFrame(result,columns=['score'])
# id = id.copy().reset_index(drop=True)
# file = pd.concat([id,final],axis=1)
# file.to_csv('res.csv',index=0,float_format='%.0f',encoding='utf-8')





