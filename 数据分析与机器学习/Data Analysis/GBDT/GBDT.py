# GBDT及其相关算法可分类识别，可预测回归，不限制数据类型

import pandas as pd
import matplotlib.pyplot as plt

default = pd.read_excel(r'C:\Users\123\Desktop\default of credit card clients.xls')
# 绘制饼图查看是否违约的客户比例是否失衡（大于9：1）
plt.axes(aspect = 'equal')
# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 统计客户是否违约的频数
counts = default.y.value_counts()
plt.pie(x = counts,          # 绘图数据
        labels=pd.Series(counts.index).map({0:'不违约',1:'违约'}),   # 添加文字标签
        autopct='%.1f%%'     # 设置百分比的格式，这里保留一位小数
       )
plt.show()

#————————————————————————————————————————————AdaBoost提升算法（可分类可回归）————————————————————————————————————————#

from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics

# 排除数据集中的ID变量和因变量，剩余的数据用作自变量X
X = default.drop(['ID','y'], axis = 1)
y = default.y
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25, random_state = 1234)

AdaBoost1 = ensemble.AdaBoostClassifier()        # 构建AdaBoost算法的类
AdaBoost1.fit(X_train,y_train)                   # 默认算法在训练数据集上的拟合
pred1 = AdaBoost1.predict(X_test)                # 默认算法在测试数据集上的预测
print('模型的准确率为：\n',metrics.accuracy_score(y_test, pred1))
print('模型的评估报告：\n',metrics.classification_report(y_test, pred1))
# 计算客户违约的概率值，用于生成ROC曲线的数据
y_score = AdaBoost1.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr,tpr)                   # 计算默认AUC的值
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()                                       # 面积0.78，需进一步改进模型

# 自变量的重要性排序（原模型拟合不理想，需分别从数据自变量重要性、参数调优两方面入手）
importance = pd.Series(AdaBoost1.feature_importances_, index = X.columns)
importance.sort_values().plot(kind = 'barh')
plt.show()
# 取出重要性比较高的自变量建模
predictors = list(importance[importance>0.01].index)

##（需调优两层参数：基础决策树的参数 & 提升树模型的参数）
# 第一步，通过网格搜索法选择基础模型所对应的合理参数组合
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
max_depth = [3,4,5,6]
params1 = {'base_estimator__max_depth':max_depth}# 注意此处为基础决策树所以max_depth前需base_estimator
base_model = GridSearchCV(estimator = ensemble.AdaBoostClassifier(base_estimator = DecisionTreeClassifier()),
                          param_grid= params1, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = 1)
base_model.fit(X_train[predictors],y_train)
base_model.best_params_, base_model.best_score_  # 返回基础决策树参数的最佳组合和对应AUC值       
# 第二步，通过网格搜索法选择提升树的合理参数组合
from sklearn.model_selection import GridSearchCV
n_estimators = [100,200,300]
learning_rate = [0.01,0.05,0.1,0.2]
params2 = {'n_estimators':n_estimators,'learning_rate':learning_rate} # 此处为提升树，故设置搜索参数不需要前缀
adaboost = GridSearchCV(estimator = ensemble.AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 3)),
                        param_grid= params2, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = 1)
adaboost.fit(X_train[predictors] ,y_train)
adaboost.best_params_, adaboost.best_score_      # 返回提升树参数的最佳组合和对应AUC值
# 第三步，基于以上两层调优，使用最佳的参数组合构建AdaBoost模型
AdaBoost2 = ensemble.AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 3),
                                       n_estimators = 100, learning_rate = 0.05)
AdaBoost2.fit(X_train[predictors],y_train)       # 参数调优算法在训练数据集上的拟合
pred2 = AdaBoost2.predict(X_test[predictors])    # 参数调优算法在测试数据集上的预测
print('模型的准确率为：\n',metrics.accuracy_score(y_test, pred2))
print('模型的评估报告：\n',metrics.classification_report(y_test, pred2))
# 计算正例的预测概率，用于生成ROC曲线的数据
y_score = AdaBoost2.predict_proba(X_test[predictors])[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr,tpr)                   # 计算最优参数AUC的值
# 绘制面积图
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()                                       # 准确率提升很微小，模型拟合不够理想，需考虑更换模型

#————————————————————————————————————————————GBDT提升算法（可分类可回归，扩展版AdaBoost算法）————————————————————————————————————————#
# 运用网格搜索法选择梯度提升树的合理参数组合
learning_rate = [0.01,0.05,0.1,0.2]
n_estimators = [100,300,500]
max_depth = [3,4,5,6]
params = {'learning_rate':learning_rate,'n_estimators':n_estimators,'max_depth':max_depth}
gbdt_grid = GridSearchCV(estimator = ensemble.GradientBoostingClassifier(),
                         param_grid= params, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = 1)
gbdt_grid.fit(X_train[predictors],y_train)       # GBDT模型拟合训练集数据
gbdt_grid.best_params_, gbdt_grid.best_score_    # 返回参数的最佳组合和对应AUC值

gbdt_best = ensemble.GradientBoostingClassifier(max_depth = 6,n_estimators = 500, learning_rate = 0.01) # 代入最优参数建GBDT模
gbdt_best.fit(X_train[predictors],y_train)       # 基于最佳参数组合的GBDT模型，对训练数据集进行拟合
pred = gbdt_best.predict(X_test[predictors])     # 基于最佳参数组合的GBDT模型，对测试数据集进行预测
print('模型的准确率为：\n',metrics.accuracy_score(y_test, pred))
print('模型的评估报告：\n',metrics.classification_report(y_test, pred))
# 计算违约客户的概率值，用于生成ROC曲线的数据
y_score = gbdt_best.predict_proba(X_test[predictors])[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr,tpr)                   # 计算AUC的值
# 绘制面积图
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()                                      

#————————————————————————————————————————————XGBoost提升算法（可分类可回归）————————————————————————————————————————#


creditcard = pd.read_csv(r'C:\Users\123\Desktop\creditcard.csv')
plt.axes(aspect = 'equal')
counts = creditcard.Class.value_counts()         # 统计交易是否为欺诈的频数
# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.pie(x = counts,                              # 绘图数据（查看数据不同类别的比例）
        labels=pd.Series(counts.index).map({0:'正常',1:'欺诈'}),     # 添加文字标签
        autopct='%.2f%%'     # 设置百分比的格式，这里保留一位小数
       )
plt.show()                   # 样本类别严重不平衡，需要通过SMOTE算法进行平衡

X = creditcard.drop(['Time','Class'], axis = 1)  # 删除自变量中的Time变量
y = creditcard.Class
# 数据拆分
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.3, random_state = 1234)
# 运用SMOTE算法实现训练数据集的平衡
from imblearn.over_sampling import SMOTE
over_samples = SMOTE(random_state=1234) 
over_samples_X,over_samples_y = over_samples.fit_sample(X_train, y_train)
print(y_train.value_counts()/len(y_train))       # 重抽样前的类别比例
print(pd.Series(over_samples_y).value_counts()/len(over_samples_y)) # 重抽样后的类别比例，现在可以构建模型了

# 根据重抽样后的数据构建XGBoost模型
import xgboost
import numpy as np

# 使用网格搜索法对模型参数进行调优
learning_rate = [0.01,0.05,0.1,0.2]
max_depth = [3,4,5,6]
params = {'learning_rate':learning_rate,'max_depth':max_depth}
XGB_grid = GridSearchCV(estimator = xgboost.XGBClassifier(),
                         param_grid= params, scoring = 'roc_auc', cv = 5, n_jobs = -1, verbose = 1)
XGB_grid.fit(over_samples_X,over_samples_y)
# 返回参数的最佳组合和对应AUC值
XGB_grid.best_params_, XBG_grid.best_score_

xgboost = xgboost.XGBClassifier()                # 构建参数调优后的XGBoost分类器
xgboost.fit(over_samples_X,over_samples_y)       # 使用重抽样后的数据，对其建模
# 将模型运用到测试数据集中
resample_pred = xgboost.predict(np.array(X_test))
# 返回模型的预测效果
print('模型的准确率为：\n',metrics.accuracy_score(y_test, resample_pred))
print('模型的评估报告：\n',metrics.classification_report(y_test, resample_pred))

# 计算欺诈交易的概率值，用于生成ROC曲线的数据
y_score = xgboost.predict_proba(np.array(X_test))[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr,tpr)                   # 计算AUC的值
# 绘制面积图
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()



