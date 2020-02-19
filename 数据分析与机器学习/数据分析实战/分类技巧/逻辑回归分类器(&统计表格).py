# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import statsmodels.api as sm   # 导入统计模块（统计模块支持.summary()生成统计数据）
import statsmodels.genmod.families.links as fm  # 允许使用指定链接函数

@hlp.timeit
def fitLogisticRegression(data):
    '''
        构造逻辑回归分类器
    '''
    ''' 
    GLM是广义线性模型，sm.families.family.<familyname>.links实现链接功能把模型与响应变量的分布关联起来
    Binomial二项分布，Gamma伽马分布，Gaussian高斯分布，InverseGaussian逆高斯分布，NegativeBinomial负二项分布，Poisson泊松分布，Tweedie分布
    '''
    logistic_classifier = sm.GLM(data[1], data[0], family=sm.families.Binomial(link=fm.logit)) # link=fm.logit二项分布的链接函数是logit

    return logistic_classifier.fit()

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter03/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 拆分训练集与测试集
train_x, train_y, test_x,  test_y, labels = hlp.split_data(csv_read, y = 'credit_application') # 一参是pd.DataFrame格式数据源，二参是因变量字段名

classifier = fitLogisticRegression((train_x, train_y))
predicted = classifier.predict(test_x)

# 指定类别
predicted = [1 if elem > 0.5 else 0 for elem in predicted]  # 高于0.5则归为1，否则为0

# print out the results
hlp.printModelSummary(test_y, predicted)

# print out the parameters
print(classifier.summary())