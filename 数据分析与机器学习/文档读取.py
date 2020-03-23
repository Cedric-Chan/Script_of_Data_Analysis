import os
import pandas as pd
import time
import numpy as np

'''
read()  ：一次性读取整个文件内容。推荐使用read(size)方法，size越大运行时间越长
readline()  ：每次读取一行内容。内存不够时使用，一般不太用
readlines()   ：一次性读取整个文件内容，并按行返回到list，方便我们遍历
'''

#————————————————————————————————————————————————————读取复制选中的数据————————————————————————————————————————————————————————————
df=pd.read_clipboard()   # 但是不可复现

#————————————————————————————————————————————————————读取部分数据————————————————————————————————————————————————————————————
# 读取限定行列
file = pd.read_csv('demo.csv',nrows=1000,usecols=['column1', 'column2', 'column3'])  # 读取前1000行、三列的数据

titanic.describe().loc['min':'max','Pclass':'Parch']  # 读取描述统计中部分行列

# 随机抽样读取数据
data_sample=pd.read_csv('data.csv',engine='python', encoding='utf-8', skiprows= lambda x: x>0 and np.random.rand() > 0.1)
#* x>0保证首行读入，rand()>0.1表示全部数据的10%作为随机样本

# 数据透视表与透视图
total_actions = fullData.pivot_table(data=dxy, values='count', index='TIME', columns='TYPE', aggfunc='count')
total_actions.plot(subplots=False, figsize=(18,6), kind='area')

# 分割为两个数据子集
len(data)
data_1=data.sample(frac=0.75, random_state=1234)   # 随机选择 75% 的记录赋值给data_1
data_2=data.drop(data_1.index)   # 剩下的其他记录赋到data_2

# 分割字符串
df[['姓','名']] = df.姓名.str.split(' ', expand=True)
df['姓'] = df.姓名.str.split(' ', expand=True)[0]  # 分割后的第一个就是所需

#————————————————————————————————————————————————————快速合并数据————————————————————————————————————————————————————
from glob import glob

work_file = sorted(glob('desktop/work/stocks*.csv')) # 返回包含所有合规文件名的列表;glob 返回的是无序文件名，要用 Python 内置的 sorted() 函数排序列表
pd.concat((pd.read_csv(file) for file in work_file), ignore_index=True) # 读取文件夹中的文件并合并,忽略旧索引、重置新索引的参数


#————————————————————————————————————————————————————读取全国天气————————————————————————————————————————————————————————————
t1 = time.time()
l = []
n=0
for file in os.walk('G:/05批量合并神器/全国空气质量汇总'):
    for table in file[2]:  # file[0]是文件夹地址，file[2]是CSV名称，file[1]为空[]
        path = file[0] + '/' + table
        data = pd.read_csv(path,header=0,encoding='utf-8',engine='python')
        n = n+1       
        l.append(data)
        print('第' + str(n) + '个表格已提取')
data_result = pd.concat(l)
data_result.to_csv('desktop/data_result.csv',index=0)
t2 = time.time()
t = t2 - t1
t = round(t,2)
print('用时' + str(t) + '秒')
print('完成！')

#————————————————————————————————————————————————————设置dataframe样式————————————————————————————————————————————————————————————
# 创建样式字符字典
format_dict={'Date':'{:%m/%d/%y}','Close':'${:.2f}','Volume':'{:,}'}  # 日期是月-日-年的格式，闭市价有美元符，交易量有千分号
data.style.format(format_dict)

# 隐藏索引，闭市价最小值用红色显示，最大值用浅绿色显示
(stocks.style.format(format_dict)
.hide_index()  # 隐藏索引
.highlight_min('Close',color='red')
.highlight_max('Close',color='lightgreen')
)

# 背景色渐变的样式
(stocks.style.format(format_dict)
.hide_index()  # 隐藏索引
.background_gradient(subset='Volume',cmap='Blues')  # 对于交易量背景色按照蓝色渐变色布景
)

# 迷你条形图样式
(stocks.style.format(format_dict)
.hide_index()  # 隐藏索引
.bar('Volume',color='lightblue',align='zero')  # 对于交易量背景色按照蓝色迷你条形图布景
.set_caption('2019年11月20日股票价格')  # 添加标题
)

#———————————————————————————————————————————————————快速预览数据集————————————————————————————————————————————————————————————
import pandas_profiling

'''
ProfileReport() 函数，这个函数支持任意 DataFrame，并生成交互式 HTML 数据报告：

    第一部分是纵览数据集，还会列出数据一些可能存在的问题；

    第二部分汇总每列数据，点击 toggle details 查看更多信息；

    第三部分显示列之间的关联热力图；

    第四部分显示数据集的前几条数据。
'''
pandas_profiling.ProfileReport(titanic)


