import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_folder = 'desktop/'
riverFlows = pd.read_csv(data_folder + 'combined_flow.csv', index_col=0, parse_dates=[0])

# 自相关函数ACF
'''
ACF体现了 t时刻 的观测值和 t+lag时刻 的观测值的关联有多强，lag是时间间隔。可以使用ACF函数计算ARIMA模型的MA（移动平均）
'''
acf = {}    # to store the results
f = {}

for col in riverFlows.columns:
    acf[col] = sm.tsa.stattools.acf(riverFlows[col])

# 偏自相关函数PACF
'''
在给定了以往延迟的条件下，PACF可以看成是对当前观测值的回归。可以使用PACF曲线得到ARIMA模型的AR（自回归）
'''
pacf = {}

for col in riverFlows.columns:
    pacf[col] = sm.tsa.stattools.pacf(riverFlows[col])

# 周期图 (谱密度)
'''
可以找到数据中的基础频率，即数据中波峰和波谷的主频率
'''
sd = {}

for col in riverFlows.columns:
    sd[col] = sm.tsa.stattools.periodogram(riverFlows[col])

# 绘图
# 改变字体大小
matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('font', size=14)

colors = ['#FF6600', '#000000', '#29407C', '#660000']
fig, ax = plt.subplots(2, 3)  # 2*3布局
fig.set_size_inches(12, 7)  # 图片大小

# 绘制美国河数据图表
ax[0, 0].plot(acf['american_flow'], colors[0])
ax[0, 1].plot(pacf['american_flow'],colors[1])
ax[0, 2].plot(sd['american_flow'],  colors[2])
ax[0, 2].yaxis.tick_right() # 谱密度的y轴坐标在右侧显示

# 绘制哥伦比亚河数据图表
ax[1, 0].plot(acf['columbia_flow'], colors[0])
ax[1, 1].plot(pacf['columbia_flow'],colors[1])
ax[1, 2].plot(sd['columbia_flow'],  colors[2])
ax[1, 2].yaxis.tick_right()

# 设置列标题
ax[0, 0].set_title('ACF')
ax[0, 1].set_title('PACF')
ax[0, 2].set_title('Spectral density')

# 设置行标题
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')

# save the chart
plt.savefig(data_folder + 'acf_pacf_sd.png', dpi=300)
'''
ACF图表有一个重复的模式，说明我们的数据是周期性的，也说明我们的处理过程是不平稳的
（平稳过程指的是，方差和联合概率不随时间变化）
PACF展示了时刻t的观测值强烈依赖于时刻t-1和t-2的观测值
分析频谱密度可知基础频率大约在29，即每29个月会重复一个基本模式。与实际认知的12个月有出入
'''

# 调用sm模块绘图
fig, ax = plt.subplots(2, 2, sharex=True)   # sharex设置了子图间共享坐标轴
fig.set_size_inches(8, 7)
# 绘制美国河数据图表
sm.graphics.tsa.plot_acf(
    riverFlows['american_flow'].squeeze(), lags=40, ax=ax[0, 0]) # squeeze()将单列的数据框转换为一个序列对象

sm.graphics.tsa.plot_pacf(
    riverFlows['american_flow'].squeeze(), lags=40, ax=ax[0, 1])

# 绘制哥伦比亚河数据图表
sm.graphics.tsa.plot_acf(
    riverFlows['columbia_flow'].squeeze(), lags=40, ax=ax[1, 0])

sm.graphics.tsa.plot_pacf(
    riverFlows['columbia_flow'].squeeze(), lags=40, ax=ax[1, 1])

# 设置行标题
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')

# save the chart
plt.savefig(data_folder + 'acf_pacf.png', dpi=300)