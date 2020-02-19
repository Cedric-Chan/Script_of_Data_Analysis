# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import sklearn.naive_bayes as nb

@hlp.timeit  # @调用脚本中的timeit函数，将计时功能内嵌入拟合模型的函数中
def fitNaiveBayes(data):
    '''
        构造朴素贝叶斯分类器
    '''
    # 构造分类器对象
    naiveBayes_classifier = nb.GaussianNB()

    # 拟合模型
    return naiveBayes_classifier.fit(data[0], data[1])

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter03/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 拆分训练集与测试集
train_x, train_y, test_x,  test_y, labels = hlp.split_data(csv_read, y = 'credit_application') # 一参是pd.DataFrame格式数据源，二参是因变量字段名
# 训练模型
classifier = fitNaiveBayes((train_x, train_y))
predicted = classifier.predict(test_x)
# 输出模型结果
hlp.printModelSummary(test_y, predicted)
