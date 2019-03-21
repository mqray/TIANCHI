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
from sklearn.preprocessing import RobustScaler,MinMaxScaler,OneHotEncoder
from sklearn.metrics import  mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
columns = ['id','auth','age','college','black','4G','use_age','lasted_pay',
           'lasted_pay_money','ave_6','bill_now','surplus','arrears','sensity',
           'chat_num','market','show_3','wanda','sam','movie','tour','gym',
           'netshopping','express','finance','vedio','airplane','subway','vistor','score']
data = pd.read_csv(r'E:\TIANCHI\MOBILE\datasets\train_dataset.csv', names=columns)
data.drop(index=0, inplace=True)
data.drop(columns='id', axis=1, inplace=True)

# feature = pd.DataFrame(data.iloc[:, :-1], dtype=np.float)
# y = pd.DataFrame(data.iloc[:,-1],dtype=np.int)

data = pd.DataFrame(data[:-1], dtype=np.float)

y = data['score']
y = pd.DataFrame(y,dtype=np.int)

data.drop(columns='score', inplace=True)

data = RobustScaler(with_scaling=True).fit_transform(data)
X_train,X_val,y_train,y_val = train_test_split(data,y,test_size=0.2,random_state=3)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train,y_train)
lgb_val = lgb.Dataset(X_val,y_val,reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l1',
    'num_leaves':50,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    # 'verbose': 0
}
model = lgb.LGBMRegressor(objective='regression',num_leaves=50,learning_rate=0.1,n_estimators=115,metric='l1',bagging_fraction=0.8,feature_fraction=0.8)
params1 = {
    'max_depth':range(3,8,1),
    'num_leaves':range(50,170,20)
}
gsea = GridSearchCV(estimator=model,param_grid=params1,cv=5,verbose=1)
gsea.fit(X_train,y_train)

# gbm = lgb.train(params,lgb_train,num_boost_round=20000,valid_sets=lgb_val,early_stopping_rounds=200)
# gbm.save_model('model.txt')
# y_pred = gbm.predict(X_val,num_iteration=gbm.best_iteration)
#
# mae_pre = mean_absolute_error(y_val,y_pred)
# score = 1/(1+mae_pre)
# print(score)
# plt.figure(figsize=(12,6))
# lgb.plot_importance(gbm,max_num_features=28)
# plt.show()
# print('feature importances:',list(gbm.feature_importance()))

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
# result = gbm.predict(test_data)
#
# final = pd.DataFrame(result,columns=['score'])
# id = id.copy().reset_index(drop=True)
# file = pd.concat([id,final],axis=1)
# file.to_csv(r'E:\TIANCHI\MOBILE\datasets\lgb_res_2.csv',index=0,float_format='%.0f',encoding='utf-8')
