所有特征全部->逻辑回归
elaspe:15.529832;train_score:0.714385;cv_score:0.683838
选出相关度排名前10的特征->逻辑回归
train_score:0.680633;cv_score:0.684788
相关度前十->决策树
train_score:0.999938;cv_score:0.525678 #速度极快，但是过拟合严重
改回全部参数->train_score:1.000000;cv_score:0.561883#这个过拟合好像更加严重