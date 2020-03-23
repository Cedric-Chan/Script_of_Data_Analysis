import pandas as pd

data3 = pd.read_excel(io=r'C:\Users\123\Desktop\datas\data3.xlsx')

# astype“方法”用于数据类型的强制转换
data3['id'] = data3['id'].astype(str)                             # 数值型转字符型
data3['custom_amt'] = data3['custom_amt'].str[1:].astype(float)   # 通过字符串的切片方法[1:]实现从字符串的第二个元素开始截断

# 字符型转日期型（to_datetime函数在format参数的调节下，可以识别任意格式的字符型日期值）
data3['order_date'] = pd.to_datetime(data3['order_date'], format = '%Y年%m月%d日')

# 对数处理
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
iris = iris['data']
iris_log = FunctionTransformer(log1p).fit_transform(iris)
print(iris_log)


# 数据透视表！<数据来源，数值域，索引行，列字段>
df=pd.pivot_table(df,values=['最高气温'],index=['天气'],columns=['风向'])
# 按指定字段查看均值
df['价格'].groupby([df['出发地'],df['目的地']]).mean()
# 按指定字段对数值字段分组汇总求均值，并按成交量降序排列
df_mean=df.drop(['商品','卖家']，axis=1).groupby('位置').mean().sort_values('成交量',ascending=False)


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

# 聚合数据以看到前十名客户的总购买量和总销售额，购买量计数(count),销售额求和(sum),根据销售额降序排列
top_10 = (df.groupby( name )[ ext price , quantity ].agg({ ext price : sum , quantity : count }).sort_values(by= ext price , ascending=False))[:10].reset_index()
top_10.rename(columns={ name : Name , ext price : Sales , quantity : Purchases }, inplace=True)  # 重命名列