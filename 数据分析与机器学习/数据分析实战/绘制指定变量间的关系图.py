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