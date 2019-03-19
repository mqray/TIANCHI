# model = LinearRegression(normalize=True)
# model = SVR(kernel='rbf',C=100,gamma=1.0)
# model = DecisionTreeRegressor()

# other = train.drop(columns='score')

# scaler = MinMaxScaler()
# scaler = scaler.fit(other)
# X = scaler.transform(other)



# Binarizer(threshold=131).fit_transform(train['ave_6'].to_frame())
# Binarizer(threshold=89).fit_transform(train['bill_now'].to_frame())
# Binarizer(threshold=139).fit_transform(train['use_age'].to_frame())
# Binarizer(threshold=1).fit_transform(train['show_3'].to_frame())
# Binarizer(threshold=32).fit_transform(train['chat_num'].to_frame())
# Binarizer(threshold=100).fit_transform(train['lasted_pay_money'].to_frame())


# res.to_list()
# y_val = y_val.to_list()
# print(type(y_val),type(res))#<class 'pandas.core.series.Series'> <class 'numpy.ndarray'>

# cnt1=0
# cnt2=0
# for i in range(0,len(res)):
#     if res[i] == y_val[i]:
#         cnt1 +=1
#     else:
#         cnt2 +=1
# print('succ:{%.2f%%}'%(100*cnt1/(cnt1+cnt2)))