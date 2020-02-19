import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()        # 导入鸢尾花数据

# 标准化原数据
standardizedData = StandardScaler().fit_transform(iris.data)

# 方法一，选择最低限度主成分的包容度
pca = PCA(.90)            # 降维后包含95%的原始信息

principalComponents = pca.fit_transform(X = standardizedData)

print(pca.n_components_)

# 方法二，选择主成分的数量
pca = PCA(n_components=2) # 降维后自变量为2个

principalComponents = pca.fit_transform(X = standardizedData)

print(pca.explained_variance_ratio_.sum())  # 检查这些主成分包含的原数据信息量



# 绘制聚类效果的散点图
new=pd.DataFrame(principalComponents)       # array转数据框
new.columns=['x_new','y_new']               # 设置新数据框字段名

sns.lmplot(x = 'x_new', y = 'y_new',markers = 'o', 
           data = new, fit_reg = False, scatter_kws = {'alpha':0.8}, legend_out = False)
plt.scatter(centers[:,2], centers[:,3], marker = '*', color = 'black', s = 130)
plt.xlabel('花瓣长度')
plt.ylabel('花瓣宽度')
plt.show()
# 增加一个辅助列，将不同的花种映射到0,1,2三种值，目的方便后面图形的对比
iris['Species_map'] = iris.Species.map({'virginica':0,'setosa':1,'versicolor':2})
# 绘制原始数据三个类别的散点图
sns.lmplot(x = 'Petal_Length', y = 'Petal_Width', hue = 'Species_map', data = iris, markers = ['^','s','o'],
           fit_reg = False, scatter_kws = {'alpha':0.8}, legend_out = False)
plt.xlabel('花瓣长度')
plt.ylabel('花瓣宽度')
plt.show()                                      # 存在一些错误分割，整体分类拟合
