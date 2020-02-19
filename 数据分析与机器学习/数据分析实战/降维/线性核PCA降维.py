'''
PCA产生的主成分是按照对方差贡献降序排列的，并且保持主成分之间是正交的（无关的）
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp
import pandas as pd
import numpy as np
import sklearn.decomposition as dc

@hlp.timeit
def reduce_PCA(x):
    '''
        使用主成分分析降维
    '''
    # 创建PCA对象
    pca = dc.PCA(n_components=3, whiten=True)  # 3个主成分，whiten缩放数据使每个特征的标准差都是1
    # 从所有特征中学习主成分
    return pca.fit(x)

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter04/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 提前自变量与因变量
x = csv_read[csv_read.columns[:-1]]
y = csv_read[csv_read.columns[-1]]

# 调用reduce_PCA()
z = reduce_PCA(x)

# 显示每个成分贡献的方差
print(z.explained_variance_ratio_)

# 总体的方差
print(np.sum(z.explained_variance_ratio_))

# 绘图
# 使用不同颜色和标记绘图
color_marker = [('r','^'),('g','o')]

file_save_params = {
    'fname': 'desktop/pca_3d_alt.png', 
    'dpi': 300,
}

hlp.plot_components(z.transform(x), y, color_marker, **file_save_params)