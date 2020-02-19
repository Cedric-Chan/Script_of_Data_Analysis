'''
ARMA(自回归移动平均)模型及其泛化——ARIMA(差分自回归移动平均)模型，是从时间序列预测未来时常用的两个模型
ARIMA模型的第一步是在估算AR和MA之前做差分
'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

matplotlib.rc('xtick', labelsize=9)
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('font', size=14)

def plot_functions(data, name):
    '''
        绘制ACF和PACF函数
    '''
    fig, ax = plt.subplots(2)
    sm.graphics.tsa.plot_acf(data, lags=18, ax=ax[0])
    sm.graphics.tsa.plot_pacf(data, lags=18, ax=ax[1])
    ax[0].set_title(name.split('_')[-1])
    ax[1].set_title('')
    ax[0].set_ylabel('ACF')
    ax[1].set_ylabel('PACF')
    plt.savefig(data_folder+'/'+name+'.png', dpi=300)

def fit_model(data, params, modelType, f, t):
    '''
        应用并绘制模型
    '''
    model = sm.tsa.ARIMA(data, params)
    # fit the model
    model_res = model.fit(maxiter=600,trend='nc',start_params=[.1] * (params[0]+params[2]), tol=1e-06)  # tol为容忍度
    # create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    e = model.geterrors(model_res.params)
    ax.plot(e, colors[3])
    chartText = '{0}: ({1}, {2}, {3})'.format(
        modelType.split('_')[0], params[0], 
        params[1], params[2])
    ax.text(0.1, 0.95, chartText, transform=ax.transAxes)
    plt.savefig(data_folder+'/'+modelType+'_errors.png',dpi=300)
    # plot the model
    plot_model(data['1950':], model_res, params, modelType, f, t)  # 数据图像从1950年开始
    plt.savefig(data_folder+'/'+modelType+'.png',dpi=300)

def plot_model(data, model, params, modelType, f, t):
    '''
        绘制模型预测值，模型的本质是预测了残差的均值
    '''
    fig, ax = plt.subplots(1, figsize=(12, 8))
    data.plot(ax=ax, color=colors[0])
    model.plot_predict(f, t, ax=ax, plot_insample=False)  # plot_predict使用估算的模型参数预测未来的观测值
    # define chart text
    chartText = '{0}: ({1}, {2}, {3})'.format(
        modelType.split('_')[0], params[0], 
        params[1], params[2])
    # and put it on the chart
    ax.text(0.1, 0.95, chartText, transform=ax.transAxes)



colors = ['#FF6600', '#000000', '#29407C', '#660000']

data_folder = 'desktop/'
riverFlows = pd.read_csv(data_folder + 'combined_flow_d.csv', index_col=0, parse_dates=[0])

# 绘制残差的自相关(ACF)与偏自相关(PACF)函数
# ACF决定MA的顺序，PACF决定AR的部分（第几个数进入蓝色区间，其后参数就是几）
plot_functions(riverFlows['american_flow_r'], 'ACF_PACF_American')   # AR(2),MA(4)
plot_functions(riverFlows['columbia_flow_r'], 'ACF_PACF_Columbia')   # AR(3),MA(2)

# 根据上图调整模型
fit_model(riverFlows['american_flow_r'], (2, 0, 4), 'ARMA_American', '1960-11-30', '1962')   # 差分参数设为0，ARIMA模型就退化为ARMA模型
fit_model(riverFlows['american_flow_r'], (2, 1, 4), 'ARIMA_American', '1960-11-30', '1962')  # ARIMA参数元组中第二个元素1描述了用于差分的滞后
plot_model(riverFlows['american_flow_r'], (2, 1, 4), 'ARIMA_American', '1960-11-30', '1962')

fit_model(riverFlows['columbia_flow_r'], (3, 0, 2), 'ARMA_Columbia', '1960-09-30', '1962')
fit_model(riverFlows['columbia_flow_r'], (3, 1, 2), 'ARIMA_Columbia', '1960-09-30', '1962')

# fit american models
fit_model(riverFlows['american_flow_r'], (3, 0, 5), 'ARMA_American', '1960-11-30', '1962')
fit_model(riverFlows['american_flow_r'], (3, 1, 5), 'ARIMA_American', '1960-11-30', '1962')

# fit colum models
fit_model(riverFlows['columbia_flow_r'], (3, 0, 2), 'ARMA_Columbia', '1960-09-30', '1962')
fit_model(riverFlows['columbia_flow_r'], (3, 1, 2), 'ARIMA_Columbia', '1960-09-30', '1962')