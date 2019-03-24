# -*- coding: utf-8 -*-
# @Time    : 2019/3/22 20:36
# @Author  : mqray
# @Site    : 
# @File    : fea.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import RobustScaler,MinMaxScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
'''
先处理类别型变量
包含：
用户实名制是否通过核实	
是否大学生客户	
是否黑名单客户	
是否4G不健康客户	
缴费用户当前是否欠费缴费	
其中是否经常逛商场的人、当月是否逛过福州仓山万达、当月是否到过福州山姆会员店可组合为shopping
当月是否看电影	、当月是否景点游览和当月是否体育场馆消费组合为休闲方式amusement
'''
train_row = pd.read_csv(r'E:\TIANCHI\MOBILE\datasets\train_dataset.csv')
test = pd.read_csv(r'E:\TIANCHI\MOBILE\datasets\test_dataset.csv')
id = test['用户编码']
id = id.to_frame(name='id')

target = train_row['信用分']
train_row.drop('信用分',axis=1,inplace=True)
row_data = train_row.append(test)
full_data = row_data.reset_index(drop=True)
# print(full_data.dtypes)
# train.drop()
# print(train['是否大学生客户'].head())
# df = pd.get_dummies(train['是否大学生客户'],prefix='college')
full_data = full_data.join(pd.get_dummies(full_data['用户实名制是否通过核实'], prefix='auth'))
full_data = full_data.join(pd.get_dummies(full_data['是否大学生客户'], prefix='college'))
full_data = full_data.join(pd.get_dummies(full_data['是否黑名单客户'], prefix='black'))
full_data = full_data.join(pd.get_dummies(full_data['是否4G不健康客户'], prefix='4G_health'))
full_data = full_data.join(pd.get_dummies(full_data['缴费用户当前是否欠费缴费'], prefix='Arrears'))
shopping = full_data['是否经常逛商场的人'] + full_data['当月是否逛过福州仓山万达'] + full_data['当月是否到过福州山姆会员店']
amusement = full_data['当月是否看电影'] + full_data['当月是否景点游览'] + full_data['当月是否体育场馆消费']
full_data = full_data.join(pd.get_dummies(shopping, prefix='shopping'))
full_data = full_data.join(pd.get_dummies(amusement, prefix='amusement'))

'''
连续型变量的处理可处理为等级
用户年龄
'''
ages = [0,20,30,40,50,60]
full_data['用户年龄层'] = np.digitize(full_data['用户年龄'], bins = ages)

id_age = [0,12,24,60,120,300]
full_data['网龄层'] = np.digitize(full_data['用户网龄（月）'], bins = id_age)

full_data['预计欠费'] = full_data['用户当月账户余额（元）'] - full_data['用户近6个月平均消费值（元）']
def user_consumption(row):
    pred = row['预计欠费']
    state_now = row['缴费用户当前是否欠费缴费']
    if (pred>0) &(state_now==0):
        return 1
    else:
        return 0
full_data['可能欠费等级'] = full_data.apply(lambda x:user_consumption(x), axis=1)
# 还可以通过用户当月账单以及半年月均消费反应客户是否有异常消费，这个应该也可以做为分析样本

def chat_nums(row):
    if row>131:
        return 4
    elif row>61:
        return 3
    elif row>37:
        return 2
    elif row>12:
        return 1
    else :return 0
full_data['通话人数层级'] = full_data.apply(lambda x:chat_nums(x['当月通话交往圈人数']), axis=1)


full_data['网购消费类次数'] = full_data['当月网购类应用使用次数'] + full_data['当月物流快递类应用使用次数']
def net_shopping(row):
    if row>934:
        return 3
    elif row>251:
        return 2
    elif row>18:
        return 1
    else:
        return 0

full_data['网购消费等级'] = full_data.apply(lambda x:net_shopping(x['网购消费类次数']), axis=1)
full_data['出行需求'] = full_data['当月旅游资讯类应用使用次数'] + full_data['当月飞机类应用使用次数'] + full_data['当月火车类应用使用次数']

def trip(row):
    if row>100:
        return 3
    elif row>10:
        return  2
    elif row>0:
        return 1
    else:
        return 0
full_data['出行需求等级'] = full_data.apply(lambda x :trip(x['出行需求']), axis=1)

def finance_label(row):
    if row>100000:
        return 5
    elif row>10000:
        return 4
    elif row>1000:
        return 3
    elif row>100:
        return 2
    elif row>0:
        return 1
    else:
        return 0
full_data['金融理财等级'] = full_data.apply(lambda x :finance_label(x['当月金融理财类应用使用总次数']), axis=1)

def emporium(row):
    if row>10:
        return 1
    else: return 0
full_data['商场出现者'] = full_data.apply(lambda x :emporium(x['是否经常逛商场的人']), axis=1)

def video_times(row):
    if row>1000000:
        return 6
    elif row>100000:
        return 5
    elif row>10000:
        return 4
    elif row>1000:
        return 3
    elif row>100:
        return 2
    elif row>0:
        return 1
    else:
        return 0
full_data['视频应用等级'] = full_data.apply(lambda x:video_times(x['当月视频播放类应用使用次数']), axis=1)




full_data.drop(['用户编码'],axis=1,inplace=True)
train_data = full_data[:50000]
test_data = full_data[50000:]
# ids = ids.to_frame(name='id')
# full_data =  RobustScaler(with_scaling=True).fit_transform(data)
X_train,X_val,y_train,y_val = train_test_split(train_data,target,test_size=0.2,random_state=3)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train,y_train)
lgb_val = lgb.Dataset(X_val,y_val,reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,lgb_train,num_boost_round=20000,valid_sets=lgb_val,early_stopping_rounds=200)
gbm.save_model('model.txt')
y_pred = gbm.predict(X_val,num_iteration=gbm.best_iteration)

mae_pre = mean_absolute_error(y_val,y_pred)
score = 1/(1+mae_pre)
print(score)


result = gbm.predict(test_data)
print(type(result))
final = pd.DataFrame(result,columns=['score'])
# id = id.copy().reset_index(drop=True)
file = pd.concat([id,final],axis=1)
file.to_csv(r'E:\TIANCHI\MOBILE\result\lgb_feature2_res.csv',index=0,float_format='%.0f',encoding='utf-8')
