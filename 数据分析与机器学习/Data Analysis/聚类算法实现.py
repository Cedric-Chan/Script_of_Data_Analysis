from sklearn import cluster
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
df=pd.read_csv('trip.csv', header=0, encoding='utf-8')
df1=df.ix[:,2:]
kmeans = KMeans(n_clusters=3, random_state=10).fit(df1)
df1['jllable']=kmeans.labels_                         # 新增类别列
df_count_type=df1.groupby('jllable').apply(np.size)
 
##各个类别的数目
df_count_type
##聚类中心
kmeans.cluster_centers_
##新的dataframe，命名为new_df ，并输出到本地，命名为new_df.csv。
new_df=df1[:]
new_df
new_df.to_csv('new_df.csv')
 
##将用于聚类的数据的特征的维度降至2维，并输出降维后的数据，形成一个dataframe名字new_pca
pca = PCA(n_components=2)
new_pca = pd.DataFrame(pca.fit_transform(new_df))
print(pca.explained_variance_ratio_.sum())  # 检查这些主成分包含的原数据信息量

##可视化
# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

d = new_pca[new_df['jllable'] == 0] 
plt.plot(d[0], d[1], 'r.')
d = new_pca[new_df['jllable'] == 1]
plt.plot(d[0], d[1], 'go')
d = new_pca[new_df['jllable'] == 2]
plt.plot(d[0], d[1], 'b*')
plt.gcf().savefig('kmeans.png')
plt.show()
 
