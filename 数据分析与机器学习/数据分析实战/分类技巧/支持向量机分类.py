# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import sklearn.svm as sv

@hlp.timeit
def fitSVM(data):
    '''
        Build the SVM classifier
    '''
    # create the classifier object
    svm = sv.SVC(kernel='linear', C=20.0)  # 常用kernel='rbf'即径向基函数，c为松弛因子的惩罚项系数

    # fit the data
    return svm.fit(data[0],data[1])

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter03/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 拆分训练集与测试集
train_x, train_y, test_x,  test_y, labels = hlp.split_data(csv_read, y = 'credit_application') # 一参是pd.DataFrame格式数据源，二参是因变量字段名
# 训练模型
classifier = fitSVM((train_x, train_y))

predicted = classifier.predict(test_x)
hlp.printModelSummary(test_y, predicted)

print(classifier.support_vectors_)