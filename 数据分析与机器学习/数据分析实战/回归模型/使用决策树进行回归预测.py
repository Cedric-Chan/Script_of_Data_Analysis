# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import sklearn.tree as sk

@hlp.timeit
def regression_cart(x,y):
    '''
        CART 回归器（决策树的模型根本不会使用不显著的自变量）
    '''
    cart = sk.DecisionTreeRegressor(min_samples_split=80,
        max_features="auto", random_state=66666, 
        max_depth=5)

    cart.fit(x,y)
    return cart

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

x     = csv_read[independent]
x_red = csv_read[independent_reduced]
y     = csv_read[dependent]

# estimate the model using all variables (without PC)
regressor = regression_cart(x,y)
print('R2: ', regressor.score(x,y))

for counter, (nm, label) \
    in enumerate(
        zip(x.columns, regressor.feature_importances_)
    ):
    print("{0}. {1}: {2}".format(counter, nm,label))   # 显示各变量的显著程度

# 可视化展示
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/123/Downloads/graphviz-2.38/bin'

sk.export_graphviz(regressor, out_file='desktop/tree.dot')

with open("desktop/tree.dot") as f:
    dot_graph = f.read()

dot=graphviz.Source(dot_graph)
dot.view()

# 只用主成分建模（主成分的问题在于无法直接看懂结果）
regressor_red = regression_cart(x_red,y)
print('R: ', regressor_red.score(x_red,y))

for counter, (nm, label) \
    in enumerate(
        zip(x_red.columns, regressor_red.feature_importances_)
    ):
    print("{0}. {1}: {2}".format(counter, nm,label))

# 可视化输出
sk.export_graphviz(regressor_red, out_file='desktop/tree_red.dot')
