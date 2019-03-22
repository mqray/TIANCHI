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

train = pd.read_csv(r'E:\TIANCHI\MOBILE\datasets\train_dataset.csv')
print(train['是否大学生客户'].head())
df = pd.get_dummies(train['是否大学生客户'],prefix='college')
train = train.join(pd.get_dummies(train['用户实名制是否通过核实'],prefix='auth'))
train = train.join(pd.get_dummies(train['是否大学生客户'],prefix='college'))
train = train.join(pd.get_dummies(train['是否黑名单客户'],prefix='black'))
train = train.join(pd.get_dummies(train['是否4G不健康客户'],prefix='4G_health'))
train = train.join(pd.get_dummies(train['缴费用户当前是否欠费缴费'],prefix='Arrears'))

train = train.join(pd.get_dummies(train['当月是否看电影'],prefix='watch_movie'))
train = train.join(pd.get_dummies(train['当月是否景点游览'],prefix='tour'))
train = train.join(pd.get_dummies(train['当月是否体育场馆消费'],prefix='gym'))

shopping = train['是否经常逛商场的人']+train['当月是否逛过福州仓山万达']+train['当月是否到过福州山姆会员店']

amusement = train['当月是否看电影']+train['当月是否景点游览']+train['当月是否体育场馆消费']
train = train.join(pd.get_dummies(shopping,prefix='shopping'))
train = train.join(pd.get_dummies(amusement,prefix='amusement'))
print(train.head())
