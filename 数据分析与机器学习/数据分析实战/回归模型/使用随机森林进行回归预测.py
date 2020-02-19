# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import numpy as np
import sklearn.ensemble as en
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import cross_val_score

@hlp.timeit
def regression_rf(x,y):
    '''
        创建随机森林回归器
    '''
    random_forest = en.RandomForestRegressor(
        min_samples_split=80, random_state=666, 
        max_depth=5, n_estimators=10)

    random_forest.fit(x,y)
    return random_forest

r_filename = 'desktop/power_plant_dataset_pc.csv'
csv_read = pd.read_csv(r_filename)

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

# estimate the model using all variables (without PC)
regressor = regression_rf(x,y)
print('R: ', regressor.score(x,y))
# 测试R方的敏感度
scores = cross_val_score(regressor, x, y, cv=100)
print('Expected R2: {0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std()**2))
# 显示变量的重要度
for counter, (nm, label) \
    in enumerate(
        zip(x.columns, regressor.feature_importances_)
    ):
    print("{0}. {1}: {2}".format(counter, nm,label))

# estimate the model using only the most important feature
features = np.nonzero(regressor.feature_importances_ > 0.001)
x_red = csv_read[['net_generation_MWh']]
regressor_red = regression_rf(x_red,y)
print('R: ', regressor_red.score(x_red,y))
# test the sensitivity of R2
scores = cross_val_score(regressor_red, x_red, y, cv=100)
print('Expected R2: {0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std()**2))
# print features importance
for counter, (nm, label) \
    in enumerate(
        zip(x_red.columns, regressor_red.feature_importances_)
    ):
    print("{0}. {1}: {2}".format(counter, nm,label))