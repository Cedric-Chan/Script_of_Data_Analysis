import pandas as pd
import numpy as np

test_data=pd.read_csv(r'desktop/train.csv')
df[df.严重度.isin(['中','高',"较高"])].head(3)  # 查看指定条件的数据
df[~df.严重度.isin(['中','高',"较高"])].head(3)  # "~"表示反选，然后查看指定条件的数据  

df.系统.unique()  # 查看指定字段的范围
df.系统.value_counts()  # 查看指定字段各类别的数量
df.系统.value_counts().nlargest(3)  # 查看指定字段前三个数量的类别
df[df.系统.isin(df.系统.value_counts().nlargest(3).index)].head()  # 返回'系统'字段下前三数量的类别的数据

monthly_sales = df.groupby([pd.Grouper(key='日期', freq='M')])['金额'].agg(['sum']).reset_index()  # 按月查看销售金额
monthly_sales['月占比'] = monthly_sales['sum']/df['金额'].sum()    # 计算占年度销售比例

# 非数值值特征数值化
test_data['education'],jnum=pd.factorize(test_data['education'])
test_data['education']=test_data['education']+1

# 连续数值划分为类别
pd.cut(titanic.Age, bins=[0,13,18,28,50,99], labels=['儿童','少年','青年','中年','老年'])

### 查询
# 查询指定的行
test_data.iloc[[0,2,4,5,7]]
# 查询指定的列
test_data[['department','region','education']].head() 
# 查询数值型/字符型的列
test_data.select_dtypes(include='number').head()
test_data.select_dtypes(include='object').head()  # 字符型
test_data.select_dtypes(include='category').head()  # 类别型
# 查询指定条件的信息
test_data[(test_data['education']==1) & (test_data['age']>=35)][['employee_id','gender','KPIs_met >80%']].head()

### 统计分析
# 空值数
test_data.isna().sum()
# 非空元素的计算
test_data['age'].count() 
# 最值
test_data['age'].min() 
# 求和
test_data['age'].sum() 
# 均值
test_data['age'].mean() 
# 分位数
test_data['age'].quantile(0.1) 
# 中位数
test_data['age'].median() 
# 众数
test_data['age'].mode() 
# 方差
test_data['age'].var() 
# 标准差
test_data['age'].std() 
# 位置索引
test_data['age'].idxmin()
test_data['age'].argmin()

# 平均绝对偏差
test_data['age'].mad()  
# 偏度
test_data['age'].skew()
# 峰度
test_data['age'].kurt()

test_data.info()
test_data.describe()

# 相关系数(全部变量两两配对)
test_data.corr()  # 默认使用的是pearson，还可以使用'kendall','spearman'
# 只关注某一个变量与其余变量的相关系数
test_data.corrwith(test_data['age'])
#数值型变量间的协方差矩阵
test_data.cov()

### 增删查改操作
# 增行
student3=pd.concat([student1,student2],ignore_index='Ture')
# 增列、改列名
pd.DataFrame(student2,columns=['Age','Heught','Name','Sex','weight','Score'])

# 删除指定行
test_data[test_data['age']<25]  # axis=0是对行操作，axis=1是对列操作
# 删除指定列
test_data.drop(['region','no_of_trainings'],axis=1)

# 修改指定位置的值
student3.loc[student3['Name']=='Liu','Height']=173

#——————————————————————————————————————————聚合操作 ！！！——————————————————————————————————————————————————————————————————————————————
data.groupby('id').complain_num.sum()  # 按 id 进行 groupby() 分组，再按 complain_num 计算每组的总投诉
data.groupby('id').complain_num.transform('sum')  # 不缩减总行数执行上述功能
data.groupby('id').complain_num.agg(['sum','coumt']).head()  # 用 agg() 方法，把多个聚合函数的列表作为该方法的参数

test_data.groupby('gender').mean()
test_data.groupby(['gender','age']).mean()  # 根据年龄和性别分组，计算其他变量的平均值
test_data.drop('age',axis=1).groupby('gender').agg([np.mean,np.median])  # 对每个分组计算多个统计量

# 多重索引
titanic.groupby(['Sex','Pclass']).Survived.mean.unstack()  # unstack可以将多重索引转化为dataframe格式


# 排序操作
series=pd.Series(np.array(np.random.randint(1,20,10))).sort_values()  # 降序为(ascending=False)

# 合并连接
stu_score1=pd.merge(left = student3, right = score, on='Name', how='inner') 
stu_score4=pd.merge(left = student3, right = score, on='Name', how='outer') 

### 缺失值的处理
# 删除法
test_data.dropna()  # dropna会删除任何含有缺失值的行
test_data.dropna(how='all')  # 删除所有字段都缺失的行
test_data.dropna(how='all',axis=1)  #  删除所有字段都缺失的列
test_data.dropna(how='any')  # 删除有行为缺失值的观测行

# 利用thresh，保留一些为nan的值
test_data.dropna(thresh=len(test_data)*0.9, axis='columns').head()   # 删除列中缺失值高于 10% 的缺失值
test_data.dropna(thresh=3)  # 保留有效值大于等于三个的观测行
test_data.dropna(thresh=3,axis=1)  # 保留有效值大于等于三个的观测列
test_data.dropna(thresh=1)  # 保留不全为空值的观测行

# 替补法
df.fillna(method='ffill')   # 前项填补缺失值
df.fillna(method='bfill')   # 后项填补缺失值

df.fillna({'x1':1,'x2':2,'x3':3})  # 使用常量填充不同的列
df.fillna({'x1':x1_median,'x2':x2_mean,'x3':x3_mean})

### 数据透视表
pd.pivot_table(test_data, values=['department','education'],columns=['gender'],aggfunc='count',margins=True).unstack()

### 反转功能
df.loc[::-1].head()   # 反转行序
df.loc[::-1].reset_index(drop=True).head()   # 反转行序，drop去除原有行索引
df.loc[:,::-1].head()  # 反转列序
