'''
通过平滑消除噪音只是技巧之一，卷积及其他滤波器可从数据中提取某些频率实现降噪
卷积就是 f函数（时间序列）和 g函数（滤波器）的交叠。卷积模糊了时间序列，因此也可看成一个平滑技巧

'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sc

data_folder = 'desktop/'
riverFlows = pd.read_csv(data_folder + 'combined_flow.csv', index_col=0, parse_dates=[0])

#———————————————————————————————————————————————————————————————————————————————移动平均滤波器，线性滤波器，高斯滤波器————————————————————————————————————————————————————————————————
MA_filter     = [1] * 12   # 由12个[1]元素组成的列表，通过卷积实现一个移动平均(MA)滤波器
linear_filter = [d * (1/12) for d in range(0,13)]    # 线性滤波器，该滤波器逐步降低输出值中旧观测值的重要性
gaussian      = [0] * 6 + list(sc.signal.gaussian(13, 2)[:7])   # 高斯函数滤波器

# 卷积 （使用convolution_filter()方法过滤数据集，再用dropna()移除缺失的观测值）
conv_ma       = riverFlows.apply(
    lambda col: sm.tsa.filters.convolution_filter(
        col, MA_filter), axis=0).dropna()

conv_linear   = riverFlows.apply(
    lambda col: sm.tsa.filters.convolution_filter(
        col, linear_filter), axis=0).dropna()

conv_gauss    = riverFlows.apply(
    lambda col: sm.tsa.filters.convolution_filter(
        col, gaussian), axis=0).dropna()

# 绘图(各滤波器的响应)
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True) 
colors = ['#FF6600', '#000000', '#29407C', '#660000']
# set the size of the figure explicitly
fig.set_size_inches(16, 7)
matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('font', size=14)
# plot the charts for american
ax[0].plot(MA_filter,     colors[0])
ax[1].plot(linear_filter, colors[1]) 
ax[2].plot(gaussian,  colors[2]) 
# set titles for columns
ax[0].set_title('MA filter')
ax[1].set_title('Linear filter')
ax[2].set_title('Gaussian filter')
ax[0].set_ylim([0,2])
# save the chart
plt.savefig(data_folder + 'filters.png', dpi=300)
plt.close()

# 卷积后对数据集进行过滤
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True) 
fig.set_size_inches(16, 7)
# plot the charts for american
ax[0, 0].plot(conv_ma['american_flow'],     colors[0])
ax[0, 1].plot(conv_linear['american_flow'], colors[1]) 
ax[0, 2].plot(conv_gauss['american_flow'],  colors[2]) 
# plot the charts for columbia
ax[1, 0].plot(conv_ma['columbia_flow'],        colors[0])
ax[1, 1].plot(conv_linear['columbia_flow'],    colors[1]) 
ax[1, 2].plot(conv_gauss['columbia_flow'],     colors[2]) 
# set titles for columns
ax[0, 0].set_title('MA via convolution')
ax[0, 1].set_title('Linear')
ax[0, 2].set_title('Gaussian')
# set titles for rows
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')
# save the chart
plt.savefig(data_folder + 'filtering.png', dpi=300)
'''
通过卷积的MA产生了和之前.rolling().mean方法相同的结果
线性滤波器从数据集中移除波峰
高斯模糊不仅减少了观测值的幅度，而且让其更平滑
'''

#————————————————————————————————————————————————————————————————————————————————————————————————特殊滤波器——————————————————————————————————————————————————————————————————————————
'''
对于经济数据或纯自然中的数据，某些特殊的滤波器可能更合适，如BK、HP与CF
'''
# BK滤波器是带通滤波器，从时间序列中移除高频和低频。bkfilter()一参是数据；二参是振动的最小周期（月度18、季度6、年度1.5）；三参是振动的最大周期（月度96）；四参决定了滤波器的超前-滞后
bkfilter = sm.tsa.filters.bkfilter(riverFlows, 18, 96, 12)
# HP滤波器通过解决一个最小化问题，将初始的时间序列拆成趋势和周期组件。hpfilter()一参是数据；二参是频率（月度129600，季度1600，年度6.25）
hpcycle, hptrend = sm.tsa.filters.hpfilter(riverFlows, 129600)
# CF滤波器将初始的时间序列拆成趋势和周期，和HP滤波器相似。cffilter()一参是数据；二参三参类似BK，指定了振动的最小最大周期；四参drift指定了是否要从数据中移除趋势
cfcycle, cftrend = sm.tsa.filters.cffilter(riverFlows, 18, 96, False)

# 绘图
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True) 
fig.set_size_inches(16, 7)
# plot the charts for american
ax[0, 0].plot(riverFlows['american_flow'],  colors[0])
ax[0, 1].plot(bkfilter['american_flow'], colors[1]) 
ax[0, 2].plot(hpcycle['american_flow'],   colors[2]) 
ax[0, 2].plot(hptrend['american_flow'],   colors[3]) 
ax[0, 3].plot(cfcycle['american_flow'],  colors[2]) 
ax[0, 3].plot(cftrend['american_flow'],  colors[3]) 
# plot the charts for columbia
ax[1, 0].plot(riverFlows['columbia_flow'],  colors[0])
ax[1, 1].plot(bkfilter['columbia_flow'], colors[1]) 
ax[1, 2].plot(hpcycle['columbia_flow'],   colors[2]) 
ax[1, 2].plot(hptrend['columbia_flow'],   colors[3]) 
ax[1, 3].plot(cfcycle['columbia_flow'],  colors[2]) 
ax[1, 3].plot(cftrend['columbia_flow'],  colors[3]) 
# set titles for columns
ax[0, 0].set_title('Original')
ax[0, 1].set_title('Baxter-King')
ax[0, 2].set_title('Hodrick-Prescott')
ax[0, 3].set_title('Christiano-Fitzgerald')
# set titles for rows
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')
# save the chart
plt.savefig(data_folder + 'filtering_alternative.png', dpi=300)
'''
BK过滤器从数据中移除幅度，并使其静止
分析HP过滤器，哥伦比亚河的长期趋势几乎恒定，而美国河的趋势一直在变。美国河的周期组件也体现了类似的模式
CF过滤器的输出证明了这一点，美国河的趋势组件比哥伦比亚河的更多变
'''
