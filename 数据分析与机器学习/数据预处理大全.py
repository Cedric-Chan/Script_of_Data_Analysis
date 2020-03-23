import pandas as pd


data3 = pd.read_excel(io=r'C:\Users\123\Desktop\datas\data3.xlsx')
data3.shape
#data.to_csv('C:\Users\123\Desktop\data.csv',columns=['商品','价格'],index=False,header=True,encoding='gbk') # 不要索引列,只要标题行的特定列,中文解码



#——————————————————————————————————————————————————数据类型的判断和转换——————————————————————————————————————————————
data3.dtypes

# astype“方法”用于数据类型的强制转换
data3['id'] = data3['id'].astype(str)                             # 数值型转字符型
data3['custom_amt'] = data3['custom_amt'].str[1:].astype(float)   # 通过字符串的切片方法[1:]实现从字符串的第二个元素开始截断

# 字符型转日期型（to_datetime函数在format参数的调节下，可以识别任意格式的字符型日期值）
data3['order_date'] = pd.to_datetime(data3['order_date'], format = '%Y年%m月%d日')

data3.head()

#——————————————————————————————————————————————————冗余数据的判断和处理——————————————————————————————————————————————
# 判断数据中是否存在重复观测
data3.duplicated().any()

# 默认情况下，对数据的所有变量进行判断
df.drop_duplicates()
# 基于部分变量对数据集进行重复值的删除
df.drop_duplicates(subset=['name','age'],inplace=True)

#——————————————————————————————————————————————————缺失值的识别和处理——————————————————————————————————————————————
# 判断一个数据集是否存在缺失观测，通常从两个方面入手，一个是变量的角度，即判断每个变量中是否包含缺失值；
# 另一个是数据行的角度，即判断每行数据中是否包含缺失值

# 判断各变量中是否存在缺失值
data.isnull().any(axis = 0)
# 各变量中缺失值的数量
data3.isnull().sum(axis = 0)
# 各变量中缺失值的比例
data3.isnull().sum(axis = 0)/data3.shape[0]

# 缺失观测的行数
data.isnull().any(axis = 1).sum()
# 缺失观测的比例
data.isnull().any(axis = 1).sum()/data.shape[0]


## 一、删除字段 -- 如删除缺失率非常高的edu变量
data3.drop(labels = 'edu', axis = 1, inplace=True)
# 删除观测，-- 如删除age变量中所对应的缺失观测的行  (labels参数需要指定待删除的行编号，借助index“方法”定位缺数据的行编号）
data_new = data.drop(labels = data.index[data['POTAFF_ln'].isnull()], axis = 0)
# 查看数据的规模
data_new.shape


## 二、替换法处理缺失值
#1
data3.fillna(value = {'gender': data3['gender'].mode()[0], # 使用性别的众数替换缺失性别
                 'age':data3['age'].mean()                 # 使用年龄的平均值替换缺失年龄
                 },
          inplace = True 
          )
# 再次查看各变量的缺失比例
data3.isnull().sum(axis = 0)

#2
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.nan, strategy = ‘mean’, axis = 0)   # 使用平均值替换空值
X = data.iloc[:, :-1].values          # 提取自变量组成矩阵
imputer = imputer.fit(X[:, 1:3])      # 填充自变量中索引为 1 和 2 的列的空值
X[:, 1:3] = imputer.transform(X[:, 1:3])

#3  前向替换与后向替换
df.fillna(method='pad')               # 使用前一个数据替换空值
df.fillna(method='bfill',limit=1)     # 使用后一个数据替换空值，limit限制最多只能替换连续的一个空值          


##  三、插补法
# 将数据拆分为两组，一是年龄缺失组，二是年龄非缺失组，后续基于非缺失值构建KNN模型，再对缺失组做预测
nomissing = titanic.loc[~titanic.Age.isnull(),]
missing = titanic.loc[titanic.Age.isnull(),]
# 导入机器学习的第三方包
from sklearn import neighbors
# 提取出所有的自变量
X = nomissing.columns[nomissing.columns != 'Age']
# 构建模型
knn = neighbors.KNeighborsRegressor()
# 模型拟合
knn.fit(nomissing[X], nomissing.Age)
# 年龄预测
pred_age = knn.predict(missing[X])

#——————————————————————————————————————————————————异常值的识别和处理——————————————————————————————————————————————

##1 箱线图法
# 计算下四分位数和上四分位
Q1 = sunspots.counts.quantile(q = 0.25)
Q3 = sunspots.counts.quantile(q = 0.75)
# 基于1.5倍的四分位差计算上下须对应的值
low_whisker = Q1 - 1.5*(Q3 - Q1)
up_whisker = Q3 + 1.5*(Q3 - Q1)
# 寻找异常点
sunspots.counts[(sunspots.counts > up_whisker) | (sunspots.counts < low_whisker)]


##2 正态分布图法
# 计算判断异常点和极端异常点的临界值
outlier_ll = pay_ratio.ratio.mean() - 2* pay_ratio.ratio.std()
outlier_ul = pay_ratio.ratio.mean() + 2* pay_ratio.ratio.std()

extreme_outlier_ll = pay_ratio.ratio.mean() - 3* pay_ratio.ratio.std()
extreme_outlier_ul = pay_ratio.ratio.mean() + 3* pay_ratio.ratio.std()
# 寻找异常点
pay_ratio.loc[(pay_ratio.ratio > outlier_ul) | (pay_ratio.ratio < outlier_ll), ['date','ratio']]
# 寻找极端异常点
pay_ratio.loc[(pay_ratio.ratio > extreme_outlier_ul) | (pay_ratio.ratio < extreme_outlier_ll), ['date','ratio']]


#——————————————————————————————————————————————————离散变量（文本类别）的处理————————————————————————————————————————————————
# 一、替换为离散数值变量
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# 二、创建哑变量
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#——————————————————————————————————————————————————————拆分训练集、测试集———————————————————————————————————————————————————
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

#—————————————————————————————————————————————————————特征缩放（同一变量量纲）————————————————————————————————————————————————
# 将所有数据缩放至同一量纲固然有好处，但缺点是，这丢失了解释每个观测样本归属于哪个变量的便捷性

from sklearn.preprocessing import StandardScaler
# 直接在数据集上进行拟合以及变换。获取对象并应用方法。
sc_X = StandardScaler()         
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# 不需要在测试集上进行拟合，只进行变换。
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


#——————————————————————————————————————————————————————————数据框的整理————————————————————————————————————————————————————

data.ix[0:3,['商品','价格']]                       # 选择指定行、列组成新数据块
data['销售额']=data['价格']*data['成交量']          # 根据已有字段创建新字段
data [data['价格']<100) & (data['成交量']>10000)]   # 根据条件过滤行

df=data.set_index('位置')   # 将某个字段作为索引（主）字段
df1=df.sort_index()         # 将数据框按索引字段进行排序输入
df2=df.set_index(['位置','卖家']).sortlevel(0)      # 根据两个字段排序，其中第一个为主字段

df.info()                   # 查看表的数据信息
df.describe()               # 查看表的描述性统计信息（离散）

# 按'位置'分组，计算'销量'列的平均值
group=df['销量'].groupby(group['位置'])
group.mean()







