'''
DBSCAN是基于密度的空间聚类，可识别出噪音
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp
import pandas as pd
import sklearn.cluster as cl
import numpy as np

@hlp.timeit
def findClusters_DBSCAN(data):
    '''
        使用DBSCAN聚类
    '''
    # 创建分类器对象
    dbscan = cl.DBSCAN(eps=1.2, min_samples=200)  # eps是两个样本邻接的最大距离，min_samples控制的是邻接关系

    return dbscan.fit(data)

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

# 聚类数据
cluster = findClusters_DBSCAN(selected)

# 评估聚类有效性
labels = cluster.labels_ + 1
centroids = hlp.getCentroids(selected, labels)

print('Number of clusters: {0}' .format(len(np.unique(labels))))

hlp.printClustersSummary(selected, labels, centroids)