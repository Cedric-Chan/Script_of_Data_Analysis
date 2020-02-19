'''
mean shift聚类不需指定聚类数，计算时间随观测数目呈二次方上升
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp
import pandas as pd
import sklearn.cluster as cl
import sklearn.metrics as mt

@hlp.timeit
def findClusters_meanShift(data):
    '''
        使用Mean Shift聚类
    '''
    bandwidth = cl.estimate_bandwidth(data,  # 首先需要estimate_bandwidth估算带宽
        quantile=0.25, n_samples=500)  # n_samples观测数目，quantile决定了从什么地方切分传入meanShift的核的样本

    # 创建分类器对象
    meanShift = cl.MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True  # 对观测值初始化，可显著加速算法
    )

    return meanShift.fit(data)

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

cluster = findClusters_meanShift(selected.as_matrix())

# 评估聚类模型
labels = cluster.labels_
centroids = cluster.cluster_centers_
hlp.printClustersSummary(selected, labels, centroids)