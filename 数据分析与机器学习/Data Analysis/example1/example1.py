import pandas as pd 
import numpy as np 
import seaborn as sns

### 第一步，数据预处理
income = pd.read_excel(r'C:/Users/123/Desktop/income.xlsx')   # 读取数据
income.apply(lambda x : np.sum(x.isnull()))                   # 查看数据集是否有缺失值(缺失三个离散变量)
income.fillna(value = {'workclass':income.workclass.mode()[0],
                              'occupation':income.occupation.mode()[0],
                              'native-country':income['native-country'].mode()[0]},inplace=True)   # 用众数代替缺失值
income.head()                                                 # 查看表头（查看表尾是.tail）

### 第二步，数探索性分析
income.describe()                                             # 数值变量的统计描述
income.describe(include =[ 'object'])                         # 离散型变量的统计描述
import matplotlib.pyplot as plt                               # 导入绘图模块
plt.style.use('ggplot')                                       # 设置绘图风格
fig, axes = plt.subplots(2, 1)                                # 设置多图形组合
# 绘制不同收入水平下的年龄核密度图
income.age[income.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[0], legend = True, linestyle = '-')
income.age[income.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[0], legend = True, linestyle = '--')
# 绘制不同收入水平下的周工作小时数和密度图
income['hours-per-week'][income.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[1], legend = True, linestyle = '-')
income['hours-per-week'][income.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[1], legend = True, linestyle = '--')
plt.show()                                                    # 显示图形
# 构造不同收入水平下各种族人数的数据
race = pd.DataFrame(income.groupby(by = ['race','income']).aggregate(np.size).loc[:,'age'])
race = race.reset_index()                                     # 重设行索引
race.rename(columns={'age':'counts'}, inplace=True)           # ？变量重命名（从age到counts）
race.sort_values(by = ['race','counts'], ascending=False, inplace=True)         # 排序
# 构造不同收入水平下各家庭关系人数的数据
relationship = pd.DataFrame(income.groupby(by = ['relationship','income']).aggregate(np.size).loc[:,'age'])
relationship = relationship.reset_index()
relationship.rename(columns={'age':'counts'}, inplace=True)
relationship.sort_values(by = ['relationship','counts'], ascending=False, inplace=True)
# 设置图框比例，并绘图
plt.figure(figsize=(9,5))
sns.barplot(x="race", y="counts", hue = 'income', data=race)
plt.show()                                                    # 相同种族下，年收入高低的人数差异
plt.figure(figsize=(9,5))
sns.barplot(x="relationship", y="counts", hue="income", data=relationship)
plt.show()                                                    # 相同成员关系下，年收入高低的人数差异

### 第三步，对离散变量重编码
for feature in income.columns:
    if income[feature].dtype=='object':
        income[feature]=pd.Categorical(income[feature]).codes # 采用“字符转数值”的方法对离散变量进行重编码
income.head()
income.drop(['education','fnlwgt'], axis = 1, inplace = True) # 删除造成信息冗余的和无意义的变量（度量2次的教育，序号）
income.head()                                                 # 至此数据清洗完成

### 第四步，拆分数据集（分别用于模型构建与模型评估，训练集与测试集占比常取75%：25%）
from sklearn.model_selection import train_test_split          # 导入数据集拆分函数
X_train, X_test, y_train, y_test = train_test_split(income.loc[:,'age':'native-country'], 
                                                    income['income'], train_size = 0.75, 
                                                    random_state = 1234)        # 拆分数据
print('训练数据集共有%d条观测' %X_train.shape[0])
print('测试数据集共有%d条观测' %X_test.shape[0])

### 第五步，默认参数的模型构建与评估（模型1，K近邻）
from sklearn.neighbors import KNeighborsClassifier            # 导入k近邻模型的类
kn = KNeighborsClassifier()                                   # 使用默认参数构建K近邻模型
kn.fit(X_train, y_train)                                      # 代入训练集参数
print(kn)                                                     # (学会解读)

kn_pred = kn.predict(X_test)                                  # 预测测试集（评估手段1-混淆矩阵法）
print(pd.crosstab(kn_pred, y_test))                           # （主对角线上即为对应类别预测正确的数量）
print('模型在训练集上的准确率%f' %kn.score(X_train,y_train))
print('模型在测试集上的准确率%f' %kn.score(X_test,y_test))      # 模型准确率（基于混淆矩阵计算而来）

from sklearn import metrics                                   # 导入模型评估模块
fpr, tpr, _ = metrics.roc_curve(y_test, kn.predict_proba(X_test)[:,1])          # 计算ROC曲线的x轴和y轴数据
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')        # 绘制ROC曲线（评估手段2-曲线下面积AUC值）
plt.stackplot(fpr, tpr, color = 'steelblue')                  # 添加阴影
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')  # 绘制参考线
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18)) # 往图中添加文本
plt.show()                                                    # AUC的值超过0.8基本就可以认为模型比较合理

### 第六步，默认参数的模型构建与评估（模型2，GBDT）
from sklearn.ensemble import GradientBoostingClassifier       # 导入GBDT模型的类
gbdt = GradientBoostingClassifier()                           # 使用默认参数构建GBDT模型
gbdt.fit(X_train, y_train)                                    # 代入训练集参数
print(gbdt)                                                   # (学会解读)

gbdt_pred = gbdt.predict(X_test)                              # 预测测试集（评估手段1-混淆矩阵法）
print(pd.crosstab(gbdt_pred, y_test))                         # （主对角线上即为对应类别预测正确的数量）
print('模型在训练集上的准确率%f' %gbdt.score(X_train,y_train))
print('模型在测试集上的准确率%f' %gbdt.score(X_test,y_test))    # 模型准确率（基于混淆矩阵计算而来）

fpr, tpr, _ = metrics.roc_curve(y_test, gbdt.predict_proba(X_test)[:,1])        # 计算ROC曲线的x轴和y轴数据
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')        # 绘制ROC曲线（评估手段2-曲线下面积AUC值）
plt.stackplot(fpr, tpr, color = 'steelblue')                  # 添加阴影
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')  # 绘制参考线
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18)) # 往图中添加文本
plt.show()  

### 第七步，网格搜索法求范围内模型一最优参数并评估（K近邻）
from sklearn.model_selection import GridSearchCV              # 导入网格搜索法的函数
k_options = list(range(1,12))                                 # 选择不同的参数（k从1—12中确定）
parameters = {'n_neighbors':k_options}                        
grid_kn = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = parameters, refit=True, cv=5, scoring='accuracy',verbose=0, n_jobs=-1)
grid_kn.fit(X_train, y_train)                                 # 代入训练集数据
print(grid_kn)
grid_kn.cv_results_, grid_kn.best_params_, grid_kn.best_score_# 结果输出

grid_kn_pred = grid_kn.predict(X_test)                        # 最优参数k近邻模型预测测试集（评估手段1-混淆矩阵法）
print(pd.crosstab(grid_kn_pred, y_test))

print('模型在训练集上的准确率%f' %grid_kn.score(X_train,y_train))
print('模型在测试集上的准确率%f' %grid_kn.score(X_test,y_test)) # 最优参数k近邻模型准确率

from sklearn import metrics                                   # 导入模型评估模块
fpr, tpr, _ = metrics.roc_curve(y_test, grid_kn.predict_proba(X_test)[:,1])     # 计算ROC曲线的x轴和y轴数据
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')        # 绘制ROC曲线（评估手段2-曲线下面积AUC值）
plt.stackplot(fpr, tpr, color = 'steelblue')                  # 添加阴影
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')  # 绘制参考线
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18)) # 往图中添加文本
plt.show()                                                    # AUC的值超过0.8基本就可以认为模型比较合理

### 第八步，网格搜索法求范围内模型二最优参数并评估（GBDT）
learning_rate_options = [0.01,0.05,0.1]                       # 设定模型学习速率的迭代范围、步长
max_depth_options = [3,5,7,9]                                 # 设定每个基础决策树的最大深度
n_estimators_options = [100,300,500]                          # 设定生成的基础决策树个数
parameters = {'learning_rate':learning_rate_options,'max_depth':max_depth_options,'n_estimators':n_estimators_options}
grid_gbdt = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = parameters, refit=True, cv=5, scoring='accuracy',verbose=0, n_jobs= -1)
grid_gbdt.fit(X_train, y_train)                               # 代入训练集数据
grid_gbdt.cv_results_, grid_gbdt.best_params_, grid_gbdt.best_score_           # 结果输出

grid_gbdt_pred = grid_gbdt.predict(X_test)                    # 最优参数GBDT模型预测测试集（评估手段1-混淆矩阵法）
print(pd.crosstab(grid_gbdt_pred, y_test))

print('模型在训练集上的准确率%f' %grid_gbdt.score(X_train,y_train))
print('模型在测试集上的准确率%f' %grid_gbdt.score(X_test,y_test))                # 最优参数GBDT模型准确率

from sklearn import metrics                                   # 导入模型评估模块
fpr, tpr, _ = metrics.roc_curve(y_test, grid_gbdt.predict_proba(X_test)[:,1])     # 计算ROC曲线的x轴和y轴数据
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')        # 绘制ROC曲线（评估手段2-曲线下面积AUC值）
plt.stackplot(fpr, tpr, color = 'steelblue')                  # 添加阴影
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')  # 绘制参考线
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18)) # 往图中添加文本
plt.show()              




