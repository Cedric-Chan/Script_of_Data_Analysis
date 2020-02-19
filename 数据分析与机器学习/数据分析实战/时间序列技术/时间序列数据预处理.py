import numpy as np
import pandas as pd
import pandas.tseries.offsets as ofst  # pandas的时间序列模块

# 文件名
files=['american.csv', 'columbia.csv']

# 文件夹位置
data_folder = 'desktop/'

# read the data
american = pd.read_csv(data_folder + files[0], 
    index_col=0, parse_dates=[0],    # parse_dates将第一列作为日期解析
    header=0, names=['','american_flow'])

columbia = pd.read_csv(data_folder + files[1], 
    index_col=0, parse_dates=[0],
    header=0, names=['','columbia_flow'])

# 连接数据集
riverFlows = american.combine_first(columbia)  # 连接合并数据集（列），combine_first之前的在前面

# 两个数据集时间不同，找到重叠部分
# 找到美国河缺失的第一个月
idx_american = riverFlows.index[riverFlows['american_flow'].apply(np.isnan)].min()

# 找到哥伦比亚河缺失的最后一个月
idx_columbia = riverFlows.index[riverFlows['columbia_flow'].apply(np.isnan)].max()

# 清理时间序列
riverFlows = riverFlows.truncate(   # truncate()可根据DatetimeIndex从DataFrame中移除数据
    before=idx_columbia + ofst.DateOffset(months=1),  # before指定了要舍弃哪个日期之前的记录，after指定了保留数据的最后一个日期
    after=idx_american - ofst.DateOffset(months=1)  # idx对象保存了至少有一列没有数据的日期的最小值和最大值
    )

# 写入csv
with open(data_folder + 'combined_flow.csv', 'w') as o:
    o.write(riverFlows.to_csv())

# 显示日期索引
print('\nIndex of riverFlows')
print(riverFlows.index)

# 选取区间内数据
print('\ncsv_read[\'1933\':\'1934-06\']')
print(riverFlows['1933':'1934-06'])

# 序列前移一个月
by_month = riverFlows.shift(1, freq='M')
print('\nShifting one month forward')
print(by_month.head(6))

by_year = riverFlows.shift(12, freq='M')
print('\nShifting one year forward')
print(by_year.head(6))

# 计算时间段的均值或总和
quarter = riverFlows.resample('Q', how='mean')  # Q为季末（3,6,9,12月末）
print('\nAveraging by quarter')
print(quarter.head(2))

# averaging by half a year
half = riverFlows.resample('6M', how='mean')  # 6M为半年末（6,12月）
print('\nAveraging by half a year')  
print(half.head(2))

# averaging by year
year = riverFlows.resample('A', how='mean')  # A为年末
print('\nAveraging by year')
print(year.head(2))

# 绘制时间序列图
import matplotlib
import matplotlib.pyplot as plt

# 改变字体大小
matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('font', size=14)

# colors 
colors = ['#FF6600', '#000000', '#29407C', '#660000']

# 月度时间序列
riverFlows.plot(title='Monthly river flows', color=colors)
plt.savefig(data_folder + 'monthly_riverFlows.png',dpi=300)
plt.close()   # 及时关闭进程可以释放内存

# 季度时间序列
quarter.plot(title='Quarterly river flows', color=colors)
plt.savefig(data_folder + 'quarterly_riverFlows.png',dpi=300)
plt.close()