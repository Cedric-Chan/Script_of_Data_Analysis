# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import numpy as np
import sklearn.decomposition as dc

def reduce_PCA(x, n):
    '''
        主成分分析法降维
    '''
    pca = dc.PCA(n_components=n, whiten=True)
    return pca.fit(x)

r_filename = 'desktop/power_plant_dataset.csv'
csv_read = pd.read_csv(r_filename)

x = csv_read[csv_read.columns[:-1]].copy()
y = csv_read[csv_read.columns[-1]]

# 生成自变量协方差矩阵
corr = x.corr()

# 检查矩阵的特征向量和特征值
w, v = np.linalg.eig(corr)  # linalg模块中的eig()方法找到特征值和特征向量，eig()方法要求输入一个方阵
print('Eigenvalues: ', w)

# 接近0的值意味着多重共线性，查找<0.01的值（<0.01为true），nonzero()返回非零元素索引
s = np.nonzero(w < 0.01)
# 找出共线变量
print('Indices of eigenvalues close to 0:', s[0])

all_columns = []
for i in s[0]:
    print('\nIndex: {0}. '.format(i))

    t = np.nonzero(abs(v[:,i]) > 0.33)   # 根据特征向量中显著大于0的元素找到共线的变量
    all_columns += list(t[0]) + [i]
    print('Collinear: ', t[0])

for i in np.unique(all_columns):
    print('Variable {0}: {1}'.format(i, x.columns[i]))

# 保留5个主成分降维（降维可以解决多重共线性）
n_components = 5
z = reduce_PCA(x, n=n_components)
pc = z.transform(x)

# 各主成分、所有主成分对方差的贡献有多大
print('\nVariance explained by each principal component: ', z.explained_variance_ratio_)
print('Total variance explained: ', np.sum(z.explained_variance_ratio_))

# 将降低的维度附加到数据集
for i in range(0, n_components):
    col_name = 'p_{0}'.format(i)
    x[col_name] = pd.Series(pc[:, i])
    
x[csv_read.columns[-1]] = y
csv_read = x

# 输出到文件
w_filename = 'desktop/power_plant_dataset_pc.csv'
with open(w_filename, 'w') as output:
    output.write(csv_read.to_csv(index=False))
