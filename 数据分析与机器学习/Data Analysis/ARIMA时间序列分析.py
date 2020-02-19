'''平稳性： 
平稳性就是要求经由样本时间序列所得到的拟合曲线在未来的一段期间内仍能顺着现有的形态“惯性”地延续下去
平稳性要求序列的均值和方差不发生明显变化(与连续的时间或日期紧密相关)

严平稳与弱平稳： 
严平稳：严平稳表示的分布不随时间的改变而改变。 
弱平稳：期望与相关系数（依赖性）不变。未来某时刻的t的值Xt就要依赖于它过去的信息，所以需要依赖性'''

import pandas as pd
import numpy as np
import statsmodels 
import seaborn as sns
import matplotlib.pylab as plt
from scipy import  stats
import matplotlib.pyplot as plt

data = pd.read_csv('confidence.csv', index_col='date', parse_dates=['date'])  # 指定date列作为索引，解析date列为日期类型
print(data.head())
# 切分为测试数据和训练数据
n_sample = data.shape[0]
n_train = int(0.95 * n_sample)+1
n_forecast = n_sample - n_train
ts_train = data.iloc[:n_train]['confidence']
ts_test = data.iloc[:n_forecast]['confidence']
# 对部分因变量作图查看时间相关性关系
data_short = data.loc['2007':'2017']
data_short.plot(figsize = (12,8))
plt.title("Consumer Sentiment")
plt.legend(bbox_to_anchor = (1.25,0.5))
sns.despine()
plt.show()

# 时间序列的差分d——将序列平稳化
data_short['diff_1'] = data_short['confidence'].diff(1)
# 1个时间间隔，一阶差分，再一次是二阶差分
data_short['diff_2'] = data_short['diff_1'].diff(1)
 
data_short= data_short.diff(1)
# 作图对比原数据、一阶、二阶差分结果
data_short.plot(subplots=True, figsize=(18, 12))
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
diff1 = data_short.diff(1)
diff1.plot(ax=ax1)
 
fig = plt.figure(figsize=(12,8))
ax2= fig.add_subplot(111)
diff2 = dta.diff(2)
diff2.plot(ax=ax2)
 
plt.show()

'''(ARIMA模型原理)

   自回归模型AR 
描述当前值与历史值之间的关系，用变量自身的历史时间数据对自身进行预测 
   自回归模型的限制:
1、自回归模型是用自身的数据进行预测 
2、必须具有平稳性 
3、必须具有相关性，如果自相关系数（φi）小于0.5，则不宜采用 
4、自回归只适用于预测与自身前期相关的现象

   移动平均模型MA 
移动平均模型关注的是自回归模型中的误差项的累加 

   I是差分模型
需要确定P和q， d是做几阶差分,一般1阶就可以了'''

# 通过ACF和PACF确定p、q的值
# 分别画出ACF(自相关)和PACF（偏自相关）图像
fig = plt.figure(figsize=(12,8))
 
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_short, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()
 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_short, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()

# 可视化结果：四个图的整合函数  (改参数之后即可直接调用)
  #3.2.可视化结果
def tsplot(y, lags=None, title='', figsize=(14, 8)):
 
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
 
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax
 
tsplot(data_short, title='Consumer Sentiment', lags=36)
plt.show()

# 建立模型——参数选择
arima200 = sm.tsa.ARIMA(ts_train, order=(2,0,0)).fit()   #(p,d,q)
#model_results = arima200.fit()
#遍历，寻找适宜的参数
import itertools

p_min = 0
d_min = 0
q_min = 0
p_max = 8
d_max = 0
q_max = 8

# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
 
for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
 
    try:
        model = sm.tsa.ARIMA(ts_train, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

# 画出热度图
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.show()

# #模型评价准则
train_results = sm.tsa.arma_order_select_ic(ts_train, ic=['aic', 'bic'], trend='nc', max_ar=4, max_ma=4)
 
print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)
# 当AIC和BIC的结果p，q值不一致时，需要我们重新审判

# 残差检验（正态性检验 & 自相关检验）
resid = model_results.resid     #赋值
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
plt.show()

# D-W检验（只使用于检验一阶自相关性）
print(sm.stats.durbin_watson(model_results.resid.values))

# 观察是否符合正态分布，QQ图
resid = model_results.resid     #残差
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

# Ljung-Box检验（白噪声检验），LB检验则是基于一系列滞后阶数，判断序列总体的相关性或者说随机性是否存在
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag')
# 最后一列Prob(>Q)，检验概率小于给定的显著性水平如0.05、0.10等就拒绝原假设，即为非白噪声
# 如果不是白噪声，那么就说明ARIMA模型也许并不是一个适合样本的模型

# 模型预测
predict_sunspots = model_results.predict('2016-06','2018-08', dynamic=True)
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = data.ix['2007':].plot(ax=ax)
predict_sunspots.plot(ax=ax)