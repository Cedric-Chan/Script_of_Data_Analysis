# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp
import pandas as pd
import sklearn.decomposition as dc

@hlp.timeit
def reduce_KernelPCA(x, **kwd_params):
    '''
        使用多种核的PCA降维
    '''
    # 创建PCA对象
    pca = dc.KernelPCA(**kwd_params)
    # 从所有特征中学习主成分
    return pca.fit(x)

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 拆分自变量因变量
x = csv_read[csv_read.columns[:-1]]
y = csv_read[csv_read.columns[-1]]

# 降维
kwd_params = {
        'kernel': 'rbf',  # 可使用多种核函数(linear, poly, rbf, sigmoid, cosine)
        'gamma': 0.33,    # gamma参数是poly和rbf核的系数
        'n_components': 3, # 学习3个主成分
        'max_iter': 1,  # arpack停止估算前的最大循环次数，避免陷入局部极值点
        'tol': 0.9,  # 容忍度，循环间的提升若低于tol则终止循环
        'eigen_solver': 'arpack'
    }

z = reduce_KernelPCA(x, **kwd_params)

# transform the dataset
x_transformed = z.transform(x)