'''SMOTE算法重点解决分类问题中类别型的因变量可能存在严重的偏倚，即类别之间的比例严重失调
   基本思想就是对少数类别样本进行分析和模拟，并将人工模拟的新样本添加到数据集中，进而使原始数据中的类别不再严重失衡。
   该算法的模拟过程采用了KNN技术构造新的少数类样本,后将新样本与原数据合成，产生新的训练集'''

'''  SMOTE(ratio=’auto’, random_state=None, k_neighbors=5, m_neighbors=10,
        out_step=0.5, kind=’regular’, svm_estimator=None, n_jobs=1)

ratio：用于指定重抽样的比例，如果指定字符型的值，可以是'minority'，表示对少数类别的样本进行抽样;'majority'表示对多数类别的样本进行抽样;
    'not minority'表示采用欠采样方法;'all'表示采用过采样方法。
    默认为’auto’，等同于’all’和’not minority’。如果指定字典型的值，其中键为各个类别标签，值为类别下的样本量；

random_state：用于指定随机数生成器的种子，默认为None,表示使用默认的随机数生成器；

k_neighbors：指定近邻个数，默认为5个；

m_neighbors：指定从近邻样本中随机挑选的样本个数，默认为10个；

kind：用于指定SMOTE算法在生成新样本时所使用的选项，默认为’regular’，表示对少数类别的样本进行随机采样，也可以是’borderline1’、’borderline2’和’svm’；

svm_estimator：用于指定SVM分类器，默认为sklearn.svm.SVC，该参数的目的是利用支持向量机分类器生成支持向量，然后再生成新的少数类别的样本'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics
from imblearn.over_sampling import SMOTE

# 读取数据
churn = pd.read_excel(r'C:\Users\Administrator\Desktop\Customer_Churn.xlsx')
churn.head()
# 删除state变量和area_code变量
churn.drop(labels=['state','area_code'], axis = 1, inplace = True)
# 将二元变量international_plan和voice_mail_plan转换为0-1哑变量
churn.international_plan = churn.international_plan.map({'no':0,'yes':1})
churn.voice_mail_plan = churn.voice_mail_plan.map({'no':0,'yes':1})
churn.head()

# 中文乱码的处理
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# 统计交易是否为欺诈的频数
counts = churn.churn.value_counts()

# 绘制饼图
plt.axes(aspect = 'equal')
plt.pie(x = counts, # 绘图数据
        labels=pd.Series(counts.index).map({'yes':'流失','no':'未流失'}), # 添加文字标签
        autopct='%.2f%%' # 设置百分比的格式，这里保留一位小数
       )
# 显示图形
plt.show()
#(两种类别的客户是失衡的，如果直接对这样的数据建模，可能会导致模型的结果不够准确)

# 用于建模的所有自变量
predictors = churn.columns[:-1]
# 数据拆分为训练集和测试集
X_train,X_test,y_train,y_test = model_selection.train_test_split(churn[predictors], churn.churn, random_state=12)
# 对训练数据集作平衡处理
over_samples = SMOTE(random_state=1234) 
over_samples_X,over_samples_y = over_samples.fit_sample(X_train, y_train)

# 重抽样前的类别比例
print(y_train.value_counts()/len(y_train))
# 重抽样后的类别比例
print(pd.Series(over_samples_y).value_counts()/len(over_samples_y))
#(经过SMOTE算法处理后，两个类别就可以达到1:1的平衡状态)

# 基于平衡数据重新构建决策树模型
dt2 = ensemble.DecisionTreeClassifier(n_estimators = 300)
dt2.fit(over_samples_X,over_samples_y)
# 模型在测试集上的预测
pred2 =dt2.predict(np.array(X_test))
# 模型的预测准确率
print(metrics.accuracy_score(y_test, pred2))
# 模型评估报告
print(metrics.classification_report(y_test, pred2))

#————————————————————————————————————————————————imbalance-learn—————————————————————————————————————————————————————
'''
该库是针对于不平衡的数据集进行的数据处理库，它是基于sklearn库开发而成的，因此使用起来跟sklearn有很多相似之处，上手非常的简单。
imblearn库主要对不平衡的数据采取欠采样、过采样、联合采样和集成采样四种采样方式'''

from imblearn.under_sampling import RandomUnderSampler  # 使用欠采样方法
from collections import Counter
from sklearn.datasets import make_classification  # 分类数据集

x,y = make_classification(n_samples=3000, n_features=2,n_informative=2,
                        n_redundant=0, n_repeated=0,n_classes=3,  # 3个分类
                        n_clusters_per_class=1, weights=[0.1,0.05,0.85], # 一个中心，数量比设置
                        class_sep=0.8,random_state=0)

print(Counter(y))  # 原始数据分类数量——2: 2539, 0: 298, 1: 163

rus=RandomUnderSampler(random_state=0)
x_resampled, y_resampled = rus.fit_resample(x,y)
print(sorted(Counter(y_resampled).items()))  # 欠采样后的数据类别——(0, 163), (1, 163), (2, 163)

plt.subplot(121)
plt.title('Before Sampling')
plt.scatter(x[:,0],x[:,1],marker='o',c=y,cmap='coolwarm')

plt.subplot(122)
plt.title('After Sampling')
plt.scatter(x_resampled[:,0],x_resampled[:,1],marker='o',c=y_resampled,cmap='coolwarm')
plt.show()