'''
K均值聚类是基于观测值之间的相似度进行聚类；决定因素就是观测值和最近一个聚类中心的欧几里得距离
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

#————————————————————————————————————————————————————————————————————————————————使用Scikit中的K均值聚类——————————————————————————————————————————————————————————————————————
import pandas as pd
import sklearn.cluster as cl
import sklearn.metrics as mt

@hlp.timeit
def findClusters_kmeans(data):
    '''
        K均值聚类数据
    '''
    # 创建分类器对象
    kmeans = cl.KMeans(
        n_clusters=4,
        n_jobs=-1,
        verbose=1,
        n_init=5  # 估算30个K聚类模型
    )

    return kmeans.fit(data)

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 选择变量
selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

# 开始聚类
cluster = findClusters_kmeans(selected)

# 评估聚类模型
labels = cluster.labels_
centroids = cluster.cluster_centers_

hlp.printClustersSummary(selected, labels, centroids)

#————————————————————————————————————————————————————————————————————————————————使用Scipy中的K均值聚类——————————————————————————————————————————————————————————————————————
import helper as hlp
import pandas as pd
import scipy.cluster.vq as vq

@hlp.timeit
def findClusters_kmeans(data):
    '''
       K均值聚类数据
    '''
    # 使用Scipy时需要先洗白数据
    data_w = vq.whiten(data)

    # 创建分类器对象
    kmeans, labels = vq.kmeans2(
        data_w,   # 清洗后的数据
        k=4,      # 4簇
        iter=30   # 最多迭代30次
    )

    return kmeans, labels

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 选择变量
selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

centroids, labels = findClusters_kmeans(selected.as_matrix())

hlp.printClustersSummary(selected, labels, centroids)

#————————————————————————————————————————————————————————————————————————————————寻找最优聚类数——————————————————————————————————————————————————————————————————————
import helper as hlp
import pandas as pd
import sklearn.cluster as cl
import numpy as np

@hlp.timeit
def findClusters_kmeans(data, no_of_clusters):
    '''
        K均值聚类数据
    '''
    # 创建分类器对象
    kmeans = cl.KMeans(
        n_clusters=no_of_clusters,
        n_jobs=-1,
        verbose=1,
        n_init=30
    )

    return kmeans.fit(data)

def findOptimalClusterNumber(
        data, 
        keep_going = 1, 
        max_iter = 30
    ):
    '''
        循环寻找Davis-Bouldin指标最小的聚类数
    '''
    # 存放DB指标的对象(目标是找到最小值，所以起始值要设大值)
    measures = [666]

    # 起始簇数
    n_clusters = 2 
    
    # 超过局部最小值的循环次数的计数器
    keep_going_cnt = 0  # 当新指标比之前最小值大，再迭代多少次（避免将局部最小值错认成全局最小值）
    stop = False   # 标识变量stop用于控制主循环的执行
    
    def checkMinimum(keep_going):
        '''
            检验是否找到了最小值
        '''
        global keep_going_cnt # 将keep_going_cnt定义为全局变量，从而可以全局追踪计数器

        # 新的指标是否比之前的都高
        if measures[-1] > np.min(measures[:-1]):
            keep_going_cnt += 1

            # 计数器是否超过允许值
            if keep_going_cnt > keep_going:
                # 最小值找到
                return True
        # 否则重置计数器，返回false
        else:
            keep_going_cnt = 0

        return False

    # 主循环 
    # 找到最小值或达到最大迭代次数时停止
    while not stop and n_clusters < (max_iter + 2):
        # 聚类处理
        cluster = findClusters_kmeans(data, n_clusters)

        # 评估聚类的有效性
        labels = cluster.labels_
        centroids = cluster.cluster_centers_

        # 存储指标   
        measures.append(
            hlp.davis_bouldin(data,labels, centroids)
        )

        # 检查是否找到最小值
        stop = checkMinimum(keep_going)

        # 增加循环
        n_clusters += 1

    # 若找到则返回最小值的索引
    return measures.index(np.min(measures)) + 1

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

selected = csv_read[['n_duration','n_nr_employed',
        'prev_ctc_outcome_success','n_euribor3m',
        'n_cons_conf_idx','n_age','month_oct',
        'n_cons_price_idx','edu_university_degree','n_pdays',
        'dow_mon','job_student','job_technician',
        'job_housemaid','edu_basic_6y']]

optimal_n_clusters = findOptimalClusterNumber(selected)
print('Optimal number of clusters: {0}'.format(optimal_n_clusters))
# 聚类数据
cluster = findClusters_kmeans(selected, optimal_n_clusters)
# 评估聚类的有效性
labels = cluster.labels_
centroids = cluster.cluster_centers_
hlp.printClustersSummary(selected, labels, centroids)

#————————————————————————————————————————————————————————————————————————————————数据可视化——————————————————————————————————————————————————————————————————————
import helper as hlp
import pandas as pd
import sklearn.cluster as cl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

@hlp.timeit
def findClusters_kmeans(data, no_of_clusters):
    '''
        K聚类数据
    '''
    # 创建分类器对象
    kmeans = cl.KMeans(
        n_clusters=no_of_clusters,
        n_jobs=-1,
        verbose=0,
        n_init=30
    )

    return kmeans.fit(data)

def plotInteractions(data, n_clusters):
    '''
        绘制变量之间的关系
    '''
    cluster = findClusters_kmeans(data, n_clusters)

    # 给数据集加上标签，便于绘制
    data['clus'] = cluster.labels_

    # 绘图准备，按clus划分
    ax = sns.pairplot(selected, hue='clus')

    # 保存图表
    ax.savefig(
        'desktop/k_means_{0}_clusters.png' .format(n_clusters)
    )

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 描述指定特征间的关系
columns = ['n_duration','n_cons_price_idx','n_euribor3m']
selected = csv_read[columns]

# 绘制关系图表
plotInteractions(selected, 17)