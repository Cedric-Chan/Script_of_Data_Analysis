# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import sklearn.neighbors as nb
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import cross_val_score

@hlp.timeit
def regression_kNN(x,y):
    '''
        构建KNN分类器
    '''
    knn = nb.KNeighborsRegressor(n_neighbors=80, algorithm='kd_tree', n_jobs=-1)
    knn.fit(x,y)
    return knn

r_filename = 'desktop/power_plant_dataset_pc.csv'
csv_read = pd.read_csv(r_filename)

# select the names of columns
dependent = csv_read.columns[-1]
independent_principal = [
    col 
    for col 
    in csv_read.columns 
    if col.startswith('p')
]

independent_significant = ['total_fuel_cons', 'total_fuel_cons_mmbtu']

x_sig = csv_read[independent_significant]
x_principal = csv_read[independent_principal]
y = csv_read[dependent]

# estimate the model using all variables (without PC)
regressor = regression_kNN(x_sig,y)
print('R2: ', regressor.score(x_sig,y))
# 测试R方的敏感度
scores = cross_val_score(regressor, x_sig, y, cv=100)
print('Expected R2: {0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std()**2))  # 期望值0.91，标准差0.03

# estimate the model using Principal Components only
regressor_principal = regression_kNN(x_principal,y)
print('R2: ', regressor_principal.score(x_principal,y))
# test the sensitivity of R2
scores = cross_val_score(regressor_principal, x_principal, y, cv=100)
print('Expected R2: {0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std()**2))  # 期望值为负，标准差极高
