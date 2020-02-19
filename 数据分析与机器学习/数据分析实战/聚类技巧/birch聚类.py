'''
BIRCH模型是利用层次方法的平衡迭代规约和聚类，可识别并移除噪音
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp
import pandas as pd
import sklearn.cluster as cl
import numpy as np

@hlp.timeit
def findClusters_Birch(data):
    '''
        使用BIRCH聚类
    '''
    # 创建分类器对象
    birch = cl.Birch(
        branching_factor=100,  # branching_factor控制父节点中最多有多少子聚类，超过此值聚类会递归拆分
        n_clusters=4,
        compute_labels=True,  # 结束时为每个观测值准备并存储标签
        copy=True   # BIRCH算法会直接丢弃离群值，copy设为TRUE可以复制原始数据
    )
    return birch.fit(data)

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

# 聚类数据
cluster = findClusters_Birch(selected)

# 评估模型有效性
labels = cluster.labels_
centroids = hlp.getCentroids(selected, labels)

hlp.printClustersSummary(selected, labels, centroids)