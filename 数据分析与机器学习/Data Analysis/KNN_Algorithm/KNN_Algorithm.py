# KNN模型可分类离散型因变量，也可预测连续型因变量，且对数据的分布特征无要求 
import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#————————————————————————————————————————————分类问题的解决——————————————————————————————————————————————————#
Knowledge = pd.read_excel(r'C:\Users\123\Desktop\Knowledge.xlsx')
Knowledge.head()                                            # 因变量为离散型数值学习程度
# 将数据集拆分为训练集和测试集
predictors = Knowledge.columns[:-1]
X_train, X_test, y_train, y_test = model_selection.train_test_split(Knowledge[predictors], Knowledge.UNS, 
                                                                    test_size = 0.25, random_state = 1234)
# 首先需要获取符合数据的理想K值
K = np.arange(1,np.ceil(np.log2(Knowledge.shape[0])))
accuracy = []                                               # 构建空的列表，用于存储平均准确率                
for k in K:
    # 使用10重交叉验证的方法，比对每一个k值下KNN模型的预测准确率
    cv_result = model_selection.cross_val_score(neighbors.KNeighborsClassifier(n_neighbors = int(k), weights = 'distance'), 
                                                X_train, y_train, cv = 10, scoring='accuracy')
    accuracy.append(cv_result.mean())
arg_max = np.argmax(np.array(accuracy))                     # 从k个平均准确率中挑选出最大值所对应的下标  

# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(K, accuracy)                                       # 绘制不同K值与平均预测准确率之间的折线图（离散型因变量根据准确率选K值）
plt.scatter(K, accuracy)                                    # 添加点图
plt.text(K[arg_max], accuracy[arg_max], '最佳k值为%s' %int(K[arg_max]))
plt.show()

# 利用最佳的近邻个数6，重新构建模型
knn_class = neighbors.KNeighborsClassifier(n_neighbors = 6, weights = 'distance')
knn_class.fit(X_train, y_train)                             # 模型拟合
predict = knn_class.predict(X_test)                         # 模型在测试数据集上的预测
cm = pd.crosstab(predict,y_test)                            # 构建混淆矩阵
cm
# 将混淆矩阵构造成数据框，并加上字段名和行名称，用于行或列的含义说明
cm = pd.DataFrame(cm) 
sns.heatmap(cm, annot = True,cmap = 'GnBu')                 # 绘制热力图
plt.xlabel(' Real Lable')
plt.ylabel(' Predict Lable')
plt.show()
# 模型整体的预测准确率
metrics.scorer.accuracy_score(predict,y_test)
# 分类模型的评估报告
print(metrics.classification_report(y_test, predict))       # 字段（预测精度，预测覆盖率，预测加权值，实际样本个数）

#————————————————————————————————————————————预测问题的解决——————————————————————————————————————————————————#
ccpp = pd.read_excel(r'C:\Users\123\Desktop\CCPP.xlsx')
ccpp.head()                                                 # 因变量为连续型数值发电量
ccpp.shape
from sklearn.preprocessing import minmax_scale
# 对所有自变量数据作标准化处理（非常重要！）
predictors = ccpp.columns[:-1]
X = minmax_scale(ccpp[predictors])
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, ccpp.PE, 
                                                                    test_size = 0.25, random_state = 1234)
# 设置待测试的不同k值
K = np.arange(1,np.ceil(np.log2(ccpp.shape[0])))
mse = []                                                    # 构建空的列表，用于存储平均MSE
for k in K:
    # 使用10重交叉验证的方法，比对每一个k值下KNN模型的计算MSE
    cv_result = model_selection.cross_val_score(neighbors.KNeighborsRegressor(n_neighbors = int(k), weights = 'distance'), 
                                                X_train, y_train, cv = 10, scoring='neg_mean_squared_error')
    mse.append((-1*cv_result).mean())

arg_min = np.array(mse).argmin()                            # 从k个平均MSE中挑选出最小值所对应的下标（因变量是连续型数值）

plt.plot(K, mse)                                            # 绘制不同K值与平均MSE之间的折线图                      
plt.scatter(K, mse)                                         # 添加点图
plt.text(K[arg_min], mse[arg_min] + 0.5, '最佳k值为%s' %int(K[arg_min]))
plt.show()	
# 重新构建模型，并将最佳的近邻个数设置为7
knn_reg = neighbors.KNeighborsRegressor(n_neighbors = 7, weights = 'distance')
knn_reg.fit(X_train, y_train)                               # 模型拟合
predict = knn_reg.predict(X_test)                           # 模型在测试集上的预测
metrics.mean_squared_error(y_test, predict)                 # 计算MSE值
# 对比真实值和实际值（前十）
pd.DataFrame({'Real':y_test,'Predict':predict}, columns=['Real','Predict']).head(10)


#————————————————————————————————————————————对比决策树处理预测问题——————————————————————————————————————————————————#
from sklearn import tree
# 预设各参数的不同选项值
max_depth = [19,21,23,25,27]
min_samples_split = [2,4,6,8]
min_samples_leaf = [2,4,8,10,12]
parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
# 网格搜索法，测试不同的参数值
grid_dtreg = model_selection.GridSearchCV(estimator = tree.DecisionTreeRegressor(), param_grid = parameters, cv=10,n_jobs=-1)
grid_dtreg.fit(X_train, y_train)                            # 模型拟合
grid_dtreg.best_params_                                     # 返回最佳组合的参数值

# 构建用于回归的决策树
CART_Reg = tree.DecisionTreeRegressor(max_depth = 21, min_samples_leaf = 10, min_samples_split = 6)
CART_Reg.fit(X_train, y_train)                              # 回归树拟合
pred = CART_Reg.predict(X_test)                             # 模型在测试集上的预测
# 计算衡量模型好坏的MSE值
metrics.mean_squared_error(y_test, pred)	                # 误差大于KNN近邻模型

#————————————————————————————————————————————对比随机森林处理预测问题——————————————————————————————————————————————————#
from sklearn import ensemble
RF = ensemble.RandomForestRegressor(n_estimators=200, random_state=1234) 
RF.fit(X_train, y_train)                                    # 随机森林拟合
RF_pred = RF.predict(X_test)                                # 模型在测试集上的预测
metrics.mean_squared_error(y_test, RF_pred)                 # 误差小于KNN
pd.DataFrame({'Real':y_test,'Predict':RF_pred}, columns=['Real','Predict']).head(10)




