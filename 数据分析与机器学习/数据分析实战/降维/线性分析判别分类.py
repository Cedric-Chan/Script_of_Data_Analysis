# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp
import pandas as pd
import mlpy as ml

@hlp.timeit
def reduce_LDA(x, y):
    '''
        使用线性判别分析降低维度
    '''
    # 创建PCA对象
    lda = ml.LDA(method='fast')

    # 从所有特征中学习主成分
    lda.learn(x, y)

    return lda

@hlp.timeit
def fitLinearSVM(data):
    '''
        Build the linear SVM classifier
    '''
    # create the classifier object
    svm = ml.LibSvm(svm_type='c_svc', 
        kernel_type='linear', C=20.0)

    # fit the data
    svm.learn(data[0],data[1])

    # return the classifier
    return svm

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# split into independent and dependent features
x = csv_read[csv_read.columns[:-1]]
y = csv_read[csv_read.columns[-1]]

# 拆分原始数据集
train_x_orig, train_y_orig, \
test_x_orig,  test_y_orig, \
labels_orig = hlp.split_data(
    csv_read, 
    y = 'credit_application'],
    x= ['n_duration','n_nr_employed','prev_ctc_outcome_success','n_euribor3m','n_cons_conf_idx','n_age','month_oct','n_cons_price_idx','edu_university_degree','n_pdays','dow_mon','job_student','job_technician','job_housemaid']
)

# 降维处理
csv_read['reduced'] = reduce_LDA(x, y).transform(x)

# 拆分降维后的数据
train_x_r, train_y_r, \
test_x_r,  test_y_r, \
labels_r = hlp.split_data(
    csv_read, 
    y = 'credit_application',
    x = ['reduced']
)

# train the models
classifier_r    = fitLinearSVM((train_x_r, train_y_r))  # 用SVM对降维后的数据分类
classifier_orig = fitLinearSVM((train_x_orig, train_y_orig))  # 用SVM对原始数据分类

# classify the unseen data
predicted_r    = classifier_r.pred(test_x_r)  # 预测降维后的测试集
predicted_orig = classifier_orig.pred(test_x_orig)  # 预测原始数据的测试集

# 评估模型
hlp.printModelSummary(test_y_r, predicted_r)
hlp.printModelSummary(test_y_orig, predicted_orig)
