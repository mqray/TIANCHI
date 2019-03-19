# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 22:46
# @Author  : mqray
# @Site    : 
# @File    : svm.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 21:29
# @Author  : mqray
# @Site    :
# @File    : lgb.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import  mean_absolute_error
from sklearn.svm import SVR

columns = ['id','auth','age','college','black','4G','use_age','lasted_pay',
           'lasted_pay_money','ave_6','bill_now','surplus','arrears','sensity',
           'chat_num','market','show_3','wanda','sam','movie','tour','gym',
           'netshopping','express','finance','vedio','airplane','subway','vistor','score']
data = pd.read_csv(r'E:\TIANCHI\MOBILE\datasets\train_dataset.csv', names=columns)
data.drop(index=0, inplace=True)
data.drop(columns='id', axis=1, inplace=True)

features = pd.DataFrame(data.iloc[:,:-1], dtype=np.float)
y = pd.DataFrame(data.iloc[:,-1],dtype=np.int)


features =  RobustScaler(with_scaling=True).fit_transform(data)
X_train,X_val,y_train,y_val = train_test_split(features,y,test_size=0.2,random_state=3)

model_svr = SVR()
model_svr.fit(X_train,y_train)
y_pred = model_svr.predict(X_val)

mae_pre = mean_absolute_error(y_val,y_pred)
score = 1/(1+mae_pre)
print(score)

# columns1 = ['id','auth','age','college','black','4G','use_age','lasted_pay',
#            'lasted_pay_money','ave_6','bill_now','surplus','arrears','sensity',
#            'chat_num','market','show_3','wanda','sam','movie','tour','gym',
#            'netshopping','express','finance','vedio','airplane','subway','vistor']
# test_data = pd.read_csv(r'E:\TIANCHI\MOBILE\datasets\test_dataset.csv',names=columns1)
# test_data.drop(index=0,inplace=True)
#
# id = test_data['id'].copy()
# test_data.drop(['id'],axis=1,inplace=True)
# test_data = pd.DataFrame(test_data,dtype=np.float)
# test_data = RobustScaler(with_scaling=True).fit_transform(test_data)
# result = model_svr.predict(test_data)
#
# final = pd.DataFrame(result,columns=['score'])
# id = id.copy().reset_index(drop=True)
# file = pd.concat([id,final],axis=1)
# file.to_csv(r'E:\TIANCHI\MOBILE\datasets\svr_res.csv',index=0,float_format='%.0f',encoding='utf-8')
