'''
层次聚类模型的目标是构建聚类的分层。聚合的做法常是自底向上，而切分的做法常为自顶向下
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp
import pandas as pd
import scipy.cluster.hierarchy as cl
import numpy as np
import pylab as pl

@hlp.timeit
def findClusters_link(data):
    '''
        使用单链接分层聚类
    '''
    # 返回链接对象
    return cl.linkage(data, method='single') # single表示根据两个聚类之间每个点的最小距离聚合聚类

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

# cluster the data
cluster = findClusters_link(selected)

# plot the clusters
fig  = pl.figure(figsize=(16,9))
ax   = fig.add_axes([0.1, 0.1, 0.8, 0.8])
dend = cl.dendrogram(cluster, truncate_mode='level', p=20)
ax.set_xticks([])
ax.set_yticks([])

# save the figure
fig.savefig(
    '../../Data/Chapter04/hierarchical_dendrogram.png',
    dpi=300
)