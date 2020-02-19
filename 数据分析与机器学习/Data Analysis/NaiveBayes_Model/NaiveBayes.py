# 贝叶斯模型专门用于解决（因变量离散）分类问题,需要输入的自变量都具有相同的特征（数值型、离散型，或者0-1二元型）
import pandas as pd
from sklearn import naive_bayes	
from sklearn import model_selection

#———————————————————————————————————连续的数值型自变量~高斯贝叶斯分类器——————————————————————————————————#

skin = pd.read_excel(r'C:\Users\123\Desktop\Skin_Segment.xlsx')
# 设置正例和负例
skin.y = skin.y.map({2:0,1:1})               # 将因变量的2设为0，因变量的1不变
skin.y.value_counts()
# 样本拆分（iloc获取数据框子集，所有行的0-3列）
X_train,X_test,y_train,y_test = model_selection.train_test_split(skin.iloc[:,:3], skin.y, 
                                                                 test_size = 0.25, random_state=1234)
gnb = naive_bayes.GaussianNB()               # 调用高斯朴素贝叶斯分类器的“类”

gnb.fit(X_train, y_train)                    # 模型拟合
gnb_pred = gnb.predict(X_test)               # 模型在测试数据集上的预测
pd.Series(gnb_pred).value_counts()           # 各类别的预测数量

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# 构建混淆矩阵
cm = pd.crosstab(gnb_pred,y_test)
sns.heatmap(cm, annot = True, cmap = 'GnBu', fmt = 'd')
plt.xlabel('Real')
plt.ylabel('Predict')
plt.show()
print('模型的准确率为：\n',metrics.accuracy_score(y_test, gnb_pred))
print('模型的评估报告：\n',metrics.classification_report(y_test, gnb_pred))
# 计算正例的预测概率，用于生成ROC曲线的数据，进一步评估模型
y_score = gnb.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr,tpr)                             # 计算AUC的值
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)                  # 添加边际线
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')     # 添加边际线
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)     # 添加文本信息
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()                                                 # AUC值为0.94，分类器非常良好

#———————————————————————————————————离散型的自变量~多项式贝叶斯分类器——————————————————————————————————#

import pandas as pd
mushrooms = pd.read_csv(r'C:\Users\123\Desktop\mushrooms.csv')
mushrooms.head()                             # 所有变量均为字符型的离散值，需要进一步因子化处理为数值类型
mushrooms.shape
# 将字符型数据作因子化处理，将其转换为整数型数据！
columns = mushrooms.columns[1:]
for column in columns:
    mushrooms[column] = pd.factorize(mushrooms[column])[0] #  factorize返回两个元素的元组：数值和字符水平，这里需要索引的是数值
mushrooms.head()

from sklearn import model_selection
# 将数据集拆分为训练集合测试集
Predictors = mushrooms.columns[1:]
X_train,X_test,y_train,y_test = model_selection.train_test_split(mushrooms[Predictors], mushrooms['type'], 
                                                                 test_size = 0.25, random_state = 10)
from sklearn import naive_bayes
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
mnb = naive_bayes.MultinomialNB()            # 构建多项式贝叶斯分类器的“类”
mnb.fit(X_train, y_train)                    # 基于训练数据集的拟合
mnb_pred = mnb.predict(X_test)               # 基于测试数据集的预测

cm = pd.crosstab(mnb_pred,y_test)            # 构建混淆矩阵
sns.heatmap(cm, annot = True, cmap = 'GnBu', fmt = 'd')
plt.xlabel('Real')
plt.ylabel('Predict')
plt.show()
# 模型的预测准确率
print('模型的准确率为：\n',metrics.accuracy_score(y_test, mnb_pred))
print('模型的评估报告：\n',metrics.classification_report(y_test, mnb_pred))		

# 计算正例的预测概率，用于生成ROC曲线的数据
y_score = mnb.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test.map({'edible':0,'poisonous':1}), y_score) # ，经因子化处理后，必须将映射后的数值型因变量传入y.test！
roc_auc = metrics.auc(fpr,tpr)               # 计算AUC的值
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()

#———————————————————————————————————0-1二元值的自变量~伯努利贝叶斯分类器（文本处理方法）——————————————————————————————————#

evaluation = pd.read_excel(r'C:\Users\123\Desktop\Contents.xlsx',sheetname=0)
evaluation.head(10)
###（一）文本数据处理
# 1：运用正则表达式，将评论中的数字和英文去除！
evaluation.Content = evaluation.Content.str.replace('[0-9a-zA-Z]','')
evaluation.head()

# 加载可对文本进行切词的结巴包
import jieba                                 
# 2：加载自定义词库(将无法正常切割的词实现正确切割)
jieba.load_userdict(r'C:\Users\123\Desktop\all_words.txt')
# 3：读入停止词（将句子中无意义的词语删除）
with open(r'C:\Users\123\Desktop\mystopwords.txt', encoding='UTF-8') as words:
    stop_words = [i.strip() for i in words.readlines()]
# 4：构造切词的自定义函数，并在切词过程中删除停止词
def cut_word(sentence):
    words = [i for i in jieba.lcut(sentence) if i not in stop_words]
    # 切完的词用空格隔开
    result = ' '.join(words)
    return(result)
# 5：对评论内容进行批量切词
words = evaluation.Content.apply(cut_word)
# 前5行内容的切词效果
words[:5]

# 6：根据以上切词结果，构造文档词条矩阵！（行为评论内容，列为切词后的词语，元素为词语出现的频次）
from sklearn.feature_extraction.text import CountVectorizer
# 7：计算每个词在各评论内容中的次数，并将稀疏度为99%以上的词删除
counts = CountVectorizer(min_df = 0.01)
# 创建文档词条矩阵
dtm_counts = counts.fit_transform(words).toarray()
columns = counts.get_feature_names()         # 矩阵的列名称

# :8：将矩阵转换为数据框--即X变量
X = pd.DataFrame(dtm_counts, columns=columns)
y = evaluation.Type                          # 因变量y设为情感标签
X.head()

###（二）模型构建
from sklearn import model_selection
from sklearn import naive_bayes
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# 将数据集拆分为训练集和测试集
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25, random_state=1)
# 构建伯努利贝叶斯分类器
bnb = naive_bayes.BernoulliNB()
bnb.fit(X_train,y_train)                     # 模型在训练数据集上的拟合
bnb_pred = bnb.predict(X_test)               # 模型在测试数据集上的预测
# 构建混淆矩阵
cm = pd.crosstab(bnb_pred,y_test)
sns.heatmap(cm, annot = True, cmap = 'GnBu', fmt = 'd')
plt.xlabel('Real')
plt.ylabel('Predict')
plt.show()
# 模型的预测准确率
print('模型的准确率为：\n',metrics.accuracy_score(y_test, bnb_pred))
print('模型的评估报告：\n',metrics.classification_report(y_test, bnb_pred))

# 计算正例Positive所对应的概率，用于生成ROC曲线的数据
y_score = bnb.predict_proba(X_test)[:,1]
fpr,tpr,threshold = metrics.roc_curve(y_test.map({'Negative':0,'Positive':1}), y_score) # 需将映射后的数值变量传入y_test!
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



