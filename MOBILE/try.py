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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
import datetime
from sklearn.tree import DecisionTreeRegressor
columns = ['id','auth','age','college','black','4G','use_age','lasted_pay',
           'lasted_pay_money','ave_6','bill_now','surplus','arrears','sensity',
           'chat_num','market','show_3','wanda','sam','movie','tour','gym',
           'netshopping','express','finance','vedio','airplane','subway','vistor','score']
train = pd.read_csv(r'E:\TIANCHI\MOBILE\train_dataset.csv',names=columns)
train.drop(index=0,inplace=True)
train.drop(columns='id',axis=1,inplace=True)


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

# Binarizer(threshold=131).fit_transform(train['ave_6'].to_frame())
# Binarizer(threshold=89).fit_transform(train['bill_now'].to_frame())
# Binarizer(threshold=139).fit_transform(train['use_age'].to_frame())
# Binarizer(threshold=1).fit_transform(train['show_3'].to_frame())
# Binarizer(threshold=32).fit_transform(train['chat_num'].to_frame())
# Binarizer(threshold=100).fit_transform(train['lasted_pay_money'].to_frame())

y = train['score']
other = train.drop(columns='score')

# scaler = MinMaxScaler()
# scaler = scaler.fit(other)
# X = scaler.transform(other)
X =  MinMaxScaler().fit_transform(other)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=3)

# model = LinearRegression(normalize=True)
# model = SVR(kernel='rbf',C=100,gamma=1.0)

model = DecisionTreeRegressor()
model.fit(X_train,y_train)
train_score = model.score(X_train,y_train)
start = datetime.datetime.now()
cv_score = model.score(X_val,y_val)
print('train_score:{0:0.6f};cv_score:{1:.6f}'.format(train_score,cv_score))

plt.clf()




