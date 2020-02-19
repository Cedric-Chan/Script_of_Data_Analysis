### 分类问题的解决（离散型因变量survive）
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt

Titanic = pd.read_csv(r'C:\Users\123\Desktop\Titanic.csv')
Titanic.head()
# 删除无意义的变量，并检查剩余自字是否含有缺失值
Titanic.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
Titanic.isnull().sum(axis = 0)
# 对Sex分组，用各组乘客的平均年龄填充各组中的缺失年龄
fillna_Titanic = []
for i in Titanic.Sex.unique():
    update = Titanic.loc[Titanic.Sex == i,].fillna(value = {'Age': Titanic.Age[Titanic.Sex == i].mean()}, inplace = False)
    fillna_Titanic.append(update)
Titanic = pd.concat(fillna_Titanic)
# 使用Embarked变量的众数填充缺失值
Titanic.fillna(value = {'Embarked':Titanic.Embarked.mode()[0]}, inplace=True)
Titanic.head()
# 将数值型的Pclass转换为类别型，否则无法对其哑变量处理
Titanic.Pclass = Titanic.Pclass.astype('category')
# 哑变量处理
dummy = pd.get_dummies(Titanic[['Sex','Embarked','Pclass']])
# 水平合并Titanic数据集和哑变量的数据集
Titanic = pd.concat([Titanic,dummy], axis = 1)
# 删除原始的Sex、Embarked和Pclass变量
Titanic.drop(['Sex','Embarked','Pclass'], inplace=True, axis = 1)
Titanic.head()
# 取出所有自变量名称
predictors = Titanic.columns[1:]
# 将数据集拆分为训练集和测试集，且测试集的比例为25%（保留要素为所有自变量，因变量）
X_train, X_test, y_train, y_test = model_selection.train_test_split(Titanic[predictors], Titanic.Survived, 
  test_size = 0.25, random_state = 1234)
# 导入网格搜索法用以选择最佳的预剪枝参数组合，从而有效防止决策树过拟合
from sklearn.model_selection import GridSearchCV
from sklearn import tree
# 预设各参数的不同选项值
max_depth = [2,3,4,5,6]                 # 最大深度 
min_samples_split = [2,4,6,8]           # （根）中间节点处能够继续分割的最小样本量
min_samples_leaf = [2,4,8,10,12]        # 叶节点的最小样本量
# 将各参数值以字典形式组织起来
parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
# 网格搜索法，测试不同的参数值
grid_dtcateg = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = parameters, cv=10,n_jobs=-1)
grid_dtcateg.fit(X_train, y_train)      # 模型拟合
grid_dtcateg.best_params_               # 返回最佳组合的参数值

from sklearn import metrics
# 用求得的最佳预剪枝参数构建分类决策树
CART_Class = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf = 4, min_samples_split=2)
decision_tree = CART_Class.fit(X_train, y_train)            # 模型对训练集拟合
pred = CART_Class.predict(X_test)                           # 模型在测试集上的预测
print('模型在测试集的预测准确率：\n',metrics.accuracy_score(y_test, pred))
# 该准确率无法体现正例和负例的覆盖率，需绘制ROC曲线进一步验证模型的预测效果
y_score = CART_Class.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
# 计算AUC的值
roc_auc = metrics.auc(fpr,tpr)
# 绘制面积图
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()  

## 逻辑图可视化！
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.externals.six import StringIO
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/123/Downloads/graphviz-2.38/bin'  
# 绘制决策树
dot_data = StringIO()
export_graphviz(
    decision_tree,
    out_file=dot_data,  
    feature_names=predictors,
    class_names=['Unsurvived','Survived'],  
    filled=True,
    rounded=True,  
    special_characters=True
)
# 决策树展现
from IPython.display import Image,display

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
img = Image(graph.create_png())              # 去123里面找！
graph.write_png("tree.png")	# 生成png文件
graph.write_jpg("tree.jpg")	# 生成jpg文件
graph.write_pdf("tree.pdf")	# 生成pdf文件


#————————————————————————————————————————————————————构建随机森林模型进行对比—————————————————————————————————————————————————————————#
from sklearn import ensemble
# 构建随机森林
RF_class = ensemble.RandomForestClassifier(n_estimators=200, random_state=1234)
RF_class.fit(X_train, y_train)               # 随机森林的拟合
RFclass_pred = RF_class.predict(X_test)      # 模型在测试集上的预测
print('模型在测试集的预测准确率：\n',metrics.accuracy_score(y_test, RFclass_pred))
# 计算绘图数据
y_score = RF_class.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr,tpr)
# 绘图
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()                                   # AUC值更高，更合理
# 利用理想的随机森林算法计算变量的重要性程度值
importance = RF_class.feature_importances_
# 构建含序列用于绘图
Impt_Series = pd.Series(importance, index = X_train.columns)
# 对序列排序绘图
Impt_Series.sort_values(ascending = True).plot('barh')
plt.show()



