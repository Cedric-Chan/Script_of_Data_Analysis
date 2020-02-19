# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import numpy as np
import sklearn.linear_model as lm

@hlp.timeit
def regression_linear(x,y):
    '''
        估算线性回归
    '''
    # 创建回归对象
    linear = lm.LinearRegression(fit_intercept=True,normalize=True, copy_X=True, n_jobs=-1)  # fit_intercept估算常数，normalize标准化数据，copy_X复制自变量，全核运行
    linear.fit(x,y)
    return linear

r_filename = 'desktop/power_plant_dataset_pc.csv'
csv_read = pd.read_csv(r_filename)

# 选择列名
dependent = csv_read.columns[-1]  # 因变量在最后一列
independent_reduced = [    # 主成分的列名（都以p开头）
    col 
    for col 
    in csv_read.columns 
    if col.startswith('p')
]

independent = [    # 自变量的列名
    col 
    for col 
    in csv_read.columns 
    if      col not in independent_reduced
        and col not in dependent
]

# 拆分自变量和因变量
x     = csv_read[independent]
x_red = csv_read[independent_reduced]
y     = csv_read[dependent]

# 使用所有变量估算模型
regressor = regression_linear(x,y)
# 打印模型总结
print('\nR^2: {0}'.format(regressor.score(x,y)))   # R方表示模型对方差的贡献度
coeff = [(nm, coeff) 
    for nm, coeff 
    in zip(x.columns, regressor.coef_)]
intercept = regressor.intercept_
print('Coefficients: ', coeff)  # 系数列表
print('Intercept', intercept)
print('Total number of variables: ', len(coeff) + 1)

# 使用主成分变量估算模型
regressor_red = regression_linear(x_red,y)
# 打印模型总结
print('\nR^2: {0}'.format(regressor_red.score(x_red,y)))
coeff = [(nm, coeff) 
    for nm, coeff 
    in zip(x_red.columns, regressor_red.coef_)]
intercept = regressor_red.intercept_
print('Coefficients: ', coeff)
print('Intercept', intercept)
print('Total number of variables: ', len(coeff) + 1)

# 移除州的影响
columns = [col for col in independent if 'state' not in col and col != 'total_fuel_cons']
x_no_state = x[columns]
# estimate the model
regressor_nm = regression_linear(x_no_state,y)
# print model summary
print('\nR^2: {0}'.format(regressor_nm.score(x_no_state,y)))
coeff = [(nm, coeff) 
    for nm, coeff 
    in zip(x_no_state.columns, regressor_nm.coef_)]
intercept = regressor_nm.intercept_
print('Coefficients: ', coeff)
print('Intercept', intercept)
print('Total number of variables: ', len(coeff) + 1)
