'''
时间序列的要求是满足其平均值、方差和自相关都不随时间变化。平稳的时间序列才可以运用ARIMA模型
意味着，有趋势和季节性的时间过程就是不平稳的，需要进一步进行处理
'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

def period_mean(data, freq):
    '''
        计算每个频率的均值
    '''
    return np.array([np.mean(data[i::freq]) for i in range(freq)])

data_folder = 'desktop/'
riverFlows = pd.read_csv(data_folder + 'combined_flow.csv', index_col=0, parse_dates=[0])

# 移除数据的趋势
de = sm.tsa.tsatools.detrend(riverFlows, order=1, axis=0)   # order指定了趋势的类型，0代表常数，1代表趋势呈线性，2代表趋势是二次型的
pd.DataFrame(de)
# 用移除趋势后的数据创建数据框
detrended = pd.DataFrame(de)# .detrend()方法返回的是numpy数组，需进一步转为数据框方便处理。加后缀_d与原始数据集混淆
detrended.columns=['american_flow_d','columbia_flow_d']
# 加入主数据框
riverFlows = riverFlows.join(detrended)

# 计算趋势（趋势就是初始值与去趋势化之后的值之间的差别）
riverFlows['american_flow_t'] = riverFlows['american_flow'] - riverFlows['american_flow_d']
riverFlows['columbia_flow_t'] = riverFlows['columbia_flow'] - riverFlows['columbia_flow_d']

# 观测值的数目与季节组件的频率（去趋势化之后，我们可以计算季节性组件）
nobs = len(riverFlows)
freq = 12   # 计算年度的季节性，每12个月重复一个模式

# 年度季节性趋势
month=[n+1 for n in range(0,12)]
for col in ['american_flow_d']:
    period_averages_A = period_mean(riverFlows[col], freq)
period_averages_A=pd.DataFrame(period_averages_A,index=month)
#period_averages_A=period_averages_A.tolist()
for col in ['columbia_flow_d']:
    period_averages_C = period_mean(riverFlows[col], freq)
period_averages_C=pd.DataFrame(period_averages_C,index=month)

period=period_averages_A.join(period_averages_C)
period.columns=['american','columbia']

fig.set_size_inches(12, 7)
plt.plot(period_averages_A,label='american',linewidth=2)
plt.plot(period_averages_C,label='columbia',linewidth=2)
plt.legend(loc='upper right')
plt.savefig(data_folder + 'seasonal.png', dpi=300)
'''
美国河看起来更为平顺，八月份左右逐渐涨到顶峰，然后越到年末越回落
哥伦比亚河一年大部分时候都很平静，到夏季涨水特别明显
'''

# 移除季节性
for col in ['american_flow_d', 'columbia_flow_d']:
    riverFlows[col[:-2]+'_s'] = np.tile(period_averages, nobs // freq + 1)[:nobs]
    riverFlows[col[:-2]+'_r'] = np.array(riverFlows[col])- np.array(riverFlows[col[:-2]+'_s'])

# save the decomposed dataset
with open(data_folder + 'combined_flow_d.csv', 'w') as o:
    o.write(riverFlows.to_csv())

# plot the data
fig, ax = plt.subplots(2, 3, sharex=False, sharey=True) 
fig.set_size_inches(12, 7)

colors = ['#FF6600', '#000000', '#29407C', '#660000']

matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('font', size=14)
# plot the charts for american
ax[0, 0].plot(riverFlows['american_flow_t'], colors[0])
ax[0, 1].plot(riverFlows['american_flow_s'], colors[1]) 
ax[0, 2].plot(riverFlows['american_flow_r'], colors[2]) 
# plot the charts for columbia
ax[1, 0].plot(riverFlows['columbia_flow_t'], colors[0])
ax[1, 1].plot(riverFlows['columbia_flow_s'], colors[1]) 
ax[1, 2].plot(riverFlows['columbia_flow_r'], colors[2]) 
# set titles for columns
ax[0, 0].set_title('Trend')
ax[0, 1].set_title('Seasonality')
ax[0, 2].set_title('Residuals')
# set titles for rows
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')
# save the chart
plt.savefig(data_folder + 'detrended.png', dpi=300)
'''
这里我们将时间序列分解成一个线性趋势、季节性组件以及残差
两条河流量都随时间增长，但增幅不明显。年度模式的季节性方差哥伦比亚河更大；对于残差，美国河的变化更为显著
'''

#——————————————————————————————————————————————————————————————————————————statsmodels的分解时间序列的方法——————————————————————————————————————————————————————
for col in riverFlows.columns:
    # 数据的季节性分解
    sd = sm.tsa.seasonal_decompose(riverFlows[col], model='a', freq=12)
    riverFlows[col + '_resid'] = sd.resid.fillna(np.mean(sd.resid))
    riverFlows[col + '_trend'] = sd.trend.fillna(np.mean(sd.trend))
    riverFlows[col + '_seas'] = sd.seasonal.fillna(np.mean(sd.seasonal))
'''
与之前相比，主要差异在于移除趋势的方式不同
前种做法假设了趋势在整个时域上是线性的（常数、线性、二次型）
此种做法使用卷积滤波以发现基础趋势
'''
# plot the data
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True) 
fig.set_size_inches(12, 7)
# plot the charts for american
ax[0, 0].plot(riverFlows['american_flow_trend'], colors[0])
ax[0, 1].plot(riverFlows['american_flow_seas'], colors[1]) 
ax[0, 2].plot(riverFlows['american_flow_resid'],  colors[2]) 
# plot the charts for columbia
ax[1, 0].plot(riverFlows['columbia_flow_trend'], colors[0])
ax[1, 1].plot(riverFlows['columbia_flow_seas'], colors[1]) 
ax[1, 2].plot(riverFlows['columbia_flow_resid'],  colors[2]) 
# set titles for columns
ax[0, 0].set_title('Trend')
ax[0, 1].set_title('Seasonality')
ax[0, 2].set_title('Residuals')
# set titles for rows
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')
# save the chart
plt.savefig(data_folder + 'decomposed.png', dpi=300)
