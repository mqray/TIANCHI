# -*- coding: utf-8 -*-
# @Time    : 2019/3/24 11:06
# @Author  : mqray
# @Site    : 
# @File    : os_try.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
train = pd.read_csv(r'E:\TIANCHI\MOBILE\datasets\train_dataset.csv')
test_data = pd.read_csv( r'E:\TIANCHI\MOBILE\datasets\test_dataset.csv')

# 先将训练数据和测试数据合并，避免部分重复代码

id = test_data['用户编码']#先将预测数据的id值保存下来
id = id.to_frame(name='id')

# 将训练数据中的信用分单独划分出来，作为训练时的target值
target = train['信用分']
# 在合并训练集数据和测试集数据之前，需要将训练集中的信用分字段删除
train.drop('信用分',axis=1,inplace=True)
# 垂直合并
row_data = train.append(test_data)
full_data = row_data.reset_index(drop=True)



# 先处理异常数据
transform_value_feature=['用户年龄','用户网龄（月）','当月通话交往圈人数','近三个月月均商场出现次数','当月网购类应用使用次数','当月物流快递类应用使用次数'
                            ,'当月金融理财类应用使用总次数','当月视频播放类应用使用次数','当月飞机类应用使用次数','当月火车类应用使用次数','当月旅游资讯类应用使用次数']
user_feature=['缴费用户最近一次缴费金额（元）','用户近6个月平均消费值（元）','用户账单当月总费用（元）','用户当月账户余额（元）']
log_features=['当月网购类应用使用次数','当月金融理财类应用使用总次数','当月物流快递类应用使用次数','当月视频播放类应用使用次数']
for col in transform_value_feature+user_feature+log_features:
    # 设定所有值的可取上限
    ulimit = np.percentile(full_data[col].values, 99.9)
    # 设定所有值的可取上限
    llimit = np.percentile(full_data[col].values, 0.1)
    full_data.loc[full_data[col] > ulimit, col] = ulimit
    full_data.loc[full_data[col] < llimit, col] = llimit
for col in log_features:
    full_data[col] = full_data[col].map(lambda x:np.log1p(x))



#用户年龄中有为0的记录
full_data.loc[['用户年龄']==0,'用户年龄']=full_data['用户年龄'].mode()

# 对某些连续数据进行分箱操作
ages = [0,10,20,25,30,35,40,45,50,55,60,80,100]
full_data['用户年龄层'] = np.digitize(full_data['用户年龄'], bins = ages)

id_age = [0,6,12,24,60,120,240,300]
full_data['网龄层'] = np.digitize(full_data['用户网龄（月）'], bins = id_age)


# 前面分析月消费金额时推测数据是在月末采集的
# 当月不欠费，并且余额充足，反映欠费的几率低
full_data['余额_月均消费'] = full_data['用户当月账户余额（元）'] - full_data['用户近6个月平均消费值（元）']
def user_consumption(row):
    pred = full_data['用户当月账户余额（元）'] - full_data['用户近6个月平均消费值（元）']
    state_now = row['缴费用户当前是否欠费缴费']
    if (pred>0) &(state_now==0):
        return 1
    else:
        return 0
full_data['可能欠费等级'] = full_data.apply(lambda x:user_consumption(x), axis=1)
full_data['充值是否够用'] = full_data['缴费用户最近一次缴费金额（元）'] - full_data['用户账单当月总费用（元）']
full_data['缴费金额是否满足月均消费'] = full_data['缴费用户最近一次缴费金额（元）'] - full_data['用户近6个月平均消费值（元）']
full_data['当月是否异常消费'] = full_data['用户账单当月总费用（元）'] - full_data['用户近6个月平均消费值（元）']

# 月均通话人数
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


# 出行类应用使用
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

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()

params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 31,
    'verbose': -1,
    'max_depth': -1,
    'reg_alpha':2.2,
    'reg_lambda':1.4,
    'nthread': 8
}

from sklearn.model_selection import KFold

cv_pred_all = 0
en_amount = 3

oof_lgb1 = np.zeros(50000)
prediction_lgb1 = np.zeros(len(test_data))

for seed in range(en_amount):
    NFOLDS = 5
    train_label = target

    kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    kf = kfold.split(full_data[:50000], train_label)

    train_data_use = full_data[:50000].drop(['用户编码', '信用分'], axis=1)
    test_data_use = full_data[50000:].drop(['用户编码'], axis=1)

    cv_pred = np.zeros(test_data.shape[0])
    valid_best_l2_all = 0

    feature_importance_df = pd.DataFrame()
    count = 0
    for i, (train_fold, validate) in enumerate(kf):
        print('fold: ', i, ' training')
        X_train, X_validate, label_train, label_validate = \
            train_data_use.iloc[train_fold, :], train_data_use.iloc[validate, :], \
            train_label[train_fold], train_label[validate]
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=dvalid, verbose_eval=-1,
                        early_stopping_rounds=500)
        cv_pred += bst.predict(test_data_use, num_iteration=bst.best_iteration)
        valid_best_l2_all += bst.best_score['valid_0']['l1']

        oof_lgb1[validate] = bst.predict(X_validate, num_iteration=bst.best_iteration)
        prediction_lgb1 += bst.predict(test_data_use, num_iteration=bst.best_iteration) / kfold.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(X_train.columns)
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = count + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        count += 1

    cv_pred /= NFOLDS
    valid_best_l2_all /= NFOLDS

    cv_pred_all += cv_pred
cv_pred_all /= en_amount
prediction_lgb1 /= en_amount
print('cv score for valid is: ', 1 / (1 + valid_best_l2_all))