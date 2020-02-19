### 预测回归问题的解决（连续型因变量肾小球指标CKD_epi_eGFR）
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.externals.six import StringIO
from sklearn import ensemble

NHANES = pd.read_excel(r'C:\Users\123\Desktop\NHANES.xlsx')
NHANES.head()                       # 数据清洗已完成，可直接用于建模
print(NHANES.shape)
# 取出自变量名称
predictors = NHANES.columns[:-1]
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(NHANES[predictors], NHANES.CKD_epi_eGFR, 
                                                                    test_size = 0.25, random_state = 1234)
# 预设各参数的不同选项值
max_depth = [10,15,18,20,21,22]        # 数据量较大，最大深度要设在20左右
min_samples_split = [2,4,6,8]
min_samples_leaf = [2,4,8]
parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
# 网格搜索法，测试不同的参数值
grid_dtreg = GridSearchCV(estimator = tree.DecisionTreeRegressor(), param_grid = parameters, cv=10)
grid_dtreg.fit(X_train, y_train)    # 模型拟合
grid_dtreg.best_params_             # 返回最佳组合的参数值
# 利用算得的最佳预设参数构建用于回归的决策树
CART_Reg = tree.DecisionTreeRegressor(max_depth =22, min_samples_leaf = 2, min_samples_split = 2)
CART_Reg.fit(X_train, y_train)      # 回归树拟合
pred = CART_Reg.predict(X_test)     # 模型在测试集上的预测
# 计算衡量模型好坏的MSE值
metrics.mean_squared_error(y_test, pred)

#————————————————————————————————————————构建随机森林算法作对比————————————————————————————————————————————————————#
RF = ensemble.RandomForestRegressor(n_estimators=200, random_state=1234)                    # 构建用于回归的随机森林
RF.fit(X_train, y_train)            # 随机森林拟合
RF_pred = RF.predict(X_test)        # 模型在测试集上的预测
# 计算随机森林模型的MSE值（几乎仅为最优参数的决策树模型MSE值的一半）
metrics.mean_squared_error(y_test, RF_pred)
# 构建变量重要性的序列
importance = pd.Series(RF.feature_importances_, index = X_train.columns)
# 排序并绘图
importance.sort_values(10).plot('barh')
plt.show()		





