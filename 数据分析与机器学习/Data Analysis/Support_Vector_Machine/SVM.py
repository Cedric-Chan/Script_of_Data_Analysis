# 可分类离散型，可预测连续型，通常比单一的分类算法有更好的准确率
# 不适合大样本；对缺失样本敏感；对核函数敏感；SVM为黑盒模型，无法解释计算结果

#————————————————————————————————————————————SVM解决分类问题——————————————————————————————————————————————#

from sklearn import svm
import pandas as pd
from sklearn import model_selection
from sklearn import metrics

letters = pd.read_csv(r'C:\Users\123\Desktop\letterdata.csv')
letters.head()
# 将数据拆分为训练集和测试集
predictors = letters.columns[1:]
X_train,X_test,y_train,y_test = model_selection.train_test_split(letters[predictors], letters.letter, 
                                                                 test_size = 0.25, random_state = 1234)
# 使用网格搜索法，选择线性可分SVM“类”中的最佳C值
C=[0.05,0.1,0.5,1,2,5]
parameters = {'C':C}
grid_linear_svc = model_selection.GridSearchCV(estimator = svm.LinearSVC(),param_grid =parameters,scoring='accuracy',cv=5,verbose =1,n_jobs=-1)
grid_linear_svc.fit(X_train,y_train)                        # 模型在训练数据集上的拟合
grid_linear_svc.best_params_, grid_linear_svc.best_score_	# 返回交叉验证后的最佳参数值(训练集最佳准确率仅69.2%)
# 模型在测试集上的预测
pred_linear_svc = grid_linear_svc.predict(X_test)
# 模型的预测准确率
metrics.accuracy_score(y_test, pred_linear_svc)             # 测试集的准确率不足72%，考虑非线性SVM模型

# 使用网格搜索法，选择非线性SVM“类”中的最佳C值
kernel=['rbf','linear','poly','sigmoid']
C=[0.1,0.5,1,2,5]
parameters = {'kernel':kernel,'C':C}
grid_svc = model_selection.GridSearchCV(estimator = svm.SVC(),param_grid =parameters,scoring='accuracy',cv=5,verbose =1,n_jobs=-1)
grid_svc.fit(X_train,y_train)                               # 模型在训练数据集上的拟合
grid_svc.best_params_, grid_svc.best_score_                 # 返回交叉验证后的最佳参数值(惩罚系数C=5,径向基核函数，训练集准确率97.34%)
# 模型在测试集上的预测
pred_svc = grid_svc.predict(X_test)
# 模型的预测准确率
metrics.accuracy_score(y_test,pred_svc)                     # 测试集的准确率接近98%，说明非线性可分SVM拟合及预测手写字母效果很好

#————————————————————————————————————————————SVM解决预测问题——————————————————————————————————————————————#

forestfires = pd.read_csv(r'C:\Users\123\Desktop\forestfires.csv')
forestfires.head()                                          # day无用，month是字符型变量，需数值化转化
forestfires.drop('day',axis = 1, inplace = True)            # 删除day变量
forestfires.month = pd.factorize(forestfires.month)[0]      # 将月份作数值化处理
forestfires.head()

# 需先对连续型因变量做分布的探索性分析，处理偏态等情况
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
# 绘制森林烧毁面积的直方图
sns.distplot(forestfires.area, bins = 50, kde = True, fit = norm, hist_kws = {'color':'steelblue'}, 
             kde_kws = {'color':'red', 'label':'Kernel Density'}, 
             fit_kws = {'color':'black','label':'Nomal', 'linestyle':'--'})
plt.legend()
plt.show()

# 从分布看，数据呈严重右偏，需将数据进行对数处理。还需对自变量进行标准化处理
from sklearn import preprocessing
import numpy as np
from sklearn import neighbors
# 对area变量作对数变换
y = np.log1p(forestfires.area)
# 将X变量作标准化处理！
predictors = forestfires.columns[:-1]
X = preprocessing.scale(forestfires[predictors])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 1234)

# 构建默认参数的SVM回归模型
svr = svm.SVR()
svr.fit(X_train,y_train)                                    # 默认模型在训练数据集上的拟合
pred_svr = svr.predict(X_test)                              # 默认模型在测试上的预测
metrics.mean_squared_error(y_test,pred_svr)                 # 计算默认模型的MSE

# 使用网格搜索法，选择SVM回归中的最佳C值、epsilon值和gamma值
epsilon = np.arange(0.1,1.5,0.2)                            # 设定损失函数中的r值范围
C= np.arange(100,1000,200)
gamma = np.arange(0.001,0.01,0.002)                         # 设定核函数中的γ参数范围
parameters = {'epsilon':epsilon,'C':C,'gamma':gamma}
grid_svr = model_selection.GridSearchCV(estimator = svm.SVR(),param_grid =parameters,
                                        scoring='neg_mean_squared_error',cv=5,verbose =1, n_jobs=-1)
grid_svr.fit(X_train,y_train)                               # 网格参数模型在训练数据集上的拟合
print(grid_svr.best_params_, grid_svr.best_score_)          # 返回交叉验证后的最佳参数值
pred_grid_svr = grid_svr.predict(X_test)                    # 最佳参数模型在测试集上的预测
metrics.mean_squared_error(y_test,pred_grid_svr)            # 计算最佳参数模型的MSE（均方误差小于默认模型，更符合数据预测要求）

