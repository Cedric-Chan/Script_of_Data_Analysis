import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

data_folder = 'desktop/'
riverFlows = pd.read_csv(data_folder + 'combined_flow.csv', index_col=0, parse_dates=[0])

#——————————————————————————————————————————————————————————————————————移动平均法，指数平滑法，log变换——————————————————————————————————————————————————————————————
ma_transform12  = riverFlows.rolling(window=12).mean  # 移动平均法
ma_transformExp = pd.DataFrame.ewm(riverFlows, span=3).mean  # 指数平均法
log_transfrom   = riverFlows.apply(np.log)  # 对数处理（过程可逆，不损失精确度）

# 绘图
matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('font', size=14)
colors = ['#FF6600', '#000000', '#29407C', '#660000']
fig, ax = plt.subplots(2, 4, sharex=True) 
fig.set_size_inches(16, 7)
# plot the charts for american
ax[0, 0].plot(riverFlows['american_flow'],     colors[0])
ax[0, 1].plot(ma_transform12()['american_flow'], colors[1]) 
ax[0, 2].plot(ma_transformExp()['american_flow'],colors[2]) 
ax[0, 3].plot(log_transfrom['american_flow'],  colors[3])
# plot the charts for columbia
ax[1, 0].plot(riverFlows['columbia_flow'],     colors[0])
ax[1, 1].plot(ma_transform12()['columbia_flow'], colors[1]) 
ax[1, 2].plot(ma_transformExp()['columbia_flow'],colors[2]) 
ax[1, 3].plot(log_transfrom['columbia_flow'],  colors[3])
# set titles for columns
ax[0, 0].set_title('Original')
ax[0, 1].set_title('MA (year)')
ax[0, 2].set_title('MA (exponential)')
ax[0, 3].set_title('Log-transform')
# set titles for rows
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')
# save the chart
plt.savefig(data_folder + 'transform.png', dpi=300)
'''
MA(移动平均法)移除了很多噪音，揭示了数据中的局部趋势
指数平滑法不如移动平均法那么天然，它只移除数据中最大的波峰，同时保留其整体形状
log变换将数据的幅度正规化，是唯一一个完全可逆的技巧
'''

#——————————————————————————————————————————————————————————————————————————————————Holt变换与差分处理————————————————————————————————————————————————————————————————————————————
def holt_transform(column, alpha):
    '''
        应用Holt变换的函数
        y(t) = alpha * x(t) + (1-alpha) y(t-1)
    '''
    # 从列创建一个数组
    original = np.array(column)
    # 确定变换的起点（初始的观测值）
    transformed = [original[0]]
    # 对其余数据应用holt变换
    for i in range(1, len(original)):
        transformed.append(original[i] * alpha + (1-alpha) * transformed[-1])   # 每个观测值的影响由alpha参数控制

    return transformed

# 运用Holt变换
ma_transformHolt = riverFlows.apply(lambda col: holt_transform(col, 0.5), axis=0)

# 差分处理，计算时刻t与前时刻观测值之差。相较于时间序列中的值本身，差分处理更关注于预测变化
difference = riverFlows - riverFlows.shift(-1)

# 绘图
fig, ax = plt.subplots(2, 3, sharex=True) 
colors = ['#FF6600', '#000000', '#29407C', '#660000']
fig.set_size_inches(12, 7)
# plot the charts for american
ax[0, 0].plot(riverFlows['american_flow'],colors[0])
ax[0, 1].plot(ma_transformHolt['american_flow'],colors[1]) 
ax[0, 2].plot(difference['american_flow'],colors[2]) 
# plot the charts for columbia
ax[1, 0].plot(riverFlows['columbia_flow'],colors[0])
ax[1, 1].plot(ma_transformHolt['columbia_flow'],colors[1]) 
ax[1, 2].plot(difference['columbia_flow'],colors[2]) 
# set titles for columns
ax[0, 0].set_title('Original')
ax[0, 1].set_title('Holt transform')
ax[0, 2].set_title('Differencing')
# set titles for rows
ax[0, 0].set_ylabel('American')
ax[1, 0].set_ylabel('Columbia')
# save the chart
plt.savefig(data_folder + 'holt_transform.png', dpi=300)
