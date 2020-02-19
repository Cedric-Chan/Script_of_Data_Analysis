'''
C均值聚类允许每个观测值属于不止一个聚类，各从属关系有一个权重且所属各聚类权重和为1
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import skfuzzy.cluster as cl
import numpy as np

@hlp.timeit
def findClusters_cmeans(data):
    '''
        使用模糊C均值聚类算法
    '''
    # 创建分类对象
    return cl.cmeans(
        data,
        c = 5,          # 聚类数
        m = 2,          # 指数因子
        
        # 中止条件
        error = 0.01,  # 与前一次循环差异小于0.01终止循环
        maxiter = 300
    )

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

# 聚类数据(centroids是聚类的坐标,u是每个观测值的成员值,u0是每个观测值的初始成员,d是最终欧几里得距离矩阵,jm是目标函数的变更历史,p是估算模型所用循环数,fpc是模糊划分系数)
centroids, u, u0, d, jm, p, fpc = findClusters_cmeans(selected.transpose())
print(u[0:10])

# assess the clusters effectiveness
labels = [np.argmax(elem) for elem in u.transpose()]  # argmax返回列表中最大元素的索引

hlp.printClustersSummary(selected, labels, centroids)