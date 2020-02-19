# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import statsmodels.api as sm

@hlp.timeit
def regression_ols(x,y):
    '''
        估算线性回归
    '''
    # 添加常数（一列1）
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    return model.fit()

r_filename = 'desktop/power_plant_dataset_pc.csv'
csv_read = pd.read_csv(r_filename)

# 选取列名
dependent = csv_read.columns[-1]
independent_reduced = [
    col 
    for col 
    in csv_read.columns 
    if col.startswith('p')
]

independent = [
    col 
    for col 
    in csv_read.columns 
    if      col not in independent_reduced
        and col not in dependent
]

x = csv_read[independent]
y = csv_read[dependent]

# 使用所有变量估算模型
regressor = regression_ols(x,y)
print(regressor.summary())

# 移除不显著的变量
significant = ['total_fuel_cons', 'total_fuel_cons_mmbtu']
x_red = x[significant]

regressor = regression_ols(x_red,y)
print(regressor.summary())