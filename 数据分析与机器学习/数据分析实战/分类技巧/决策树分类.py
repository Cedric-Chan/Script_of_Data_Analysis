# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import sklearn.tree as sk

@hlp.timeit
def fitDecisionTree(data):
    '''
        Build a decision tree classifier
    '''
    # create the classifier object
    tree = sk.DecisionTreeClassifier(min_samples_split=1000)  # 每个决策节点的观测值不少于1000个

    # fit the data
    return tree.fit(data[0],data[1])

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter03/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# split the data into training and testing
train_x, train_y, \
test_x,  test_y, \
labels = hlp.split_data(
    csv_read, 
    y = 'credit_application',
    x = ['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']
)

# train the model
classifier = fitDecisionTree((train_x, train_y))

# classify the unseen data
predicted = classifier.predict(test_x)

# print out the results
hlp.printModelSummary(test_y, predicted)

# 变量权重
for counter, (nm, label) \
    in enumerate(
        zip(labels, classifier.feature_importances_)
    ):
    print("{0}. {1}: {2}".format(counter, nm,label))

#—————————————————————————————————————————————————————————————————生成树状图———————————————————————————————————————————————————————————————————————————————
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/123/Downloads/graphviz-2.38/bin'

with open("desktop/tree.dot") as f:
    dot_graph = f.read()

dot=graphviz.Source(dot_graph)
dot.view()