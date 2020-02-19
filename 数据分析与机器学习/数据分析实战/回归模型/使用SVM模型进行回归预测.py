# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd
import numpy as np
import sklearn.svm as sv
import matplotlib.pyplot as plt

@hlp.timeit
def regression_svm(x, y, **kw_params):
    '''
        创建SVM回归器
    '''
    svm = sv.SVR(**kw_params)
    svm.fit(x,y)
    return svm

# 生成模拟数据
x = np.arange(-2, 2, 0.004)
errors = np.random.normal(0, 0.5, size=len(x))
y = 0.8 * x**4 - 2 * x**2 +  errors

# 重塑x数组，改为列的形式
x_reg = x.reshape(-1, 1)

models_to_test = [   # 使用4种核测试
    {'kernel': 'linear'}, 
    {'kernel': 'poly','gamma': 0.5, 'C': 0.5, 'degree': 4}, 
    {'kernel': 'poly','gamma': 0.5, 'C': 0.5, 'degree': 6}, 
    {'kernel': 'rbf','gamma': 0.5, 'C': 0.5}
]

# 生成图表，具象化模型的预测
plt.figure(figsize=(len(models_to_test) * 2 + 3, 9.5))
plt.subplots_adjust(left=.05, right=.95, bottom=.05, top=.96, wspace=.1, hspace=.15)

for i, model in enumerate(models_to_test):
    regressor = regression_svm(x_reg, y, **model)
    score = regressor.score(x_reg, y)

    plt.subplot(2, 2, i + 1)  # 将当前循环次数加一
    if model['kernel'] == 'poly':
        plt.title('Kernel: {0}, deg: {1}'.format(model['kernel'], model['degree']))
    else:
        plt.title('Kernel: {0}'.format(model['kernel']))
    plt.ylim([-4, 8])
    plt.scatter(x, y)
    plt.plot(x, regressor.predict(x_reg), color='r')
    plt.text(.9, .9, ('R^2: {0:.2f}'.format(score)),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')

plt.savefig('desktop/regression_svm.png',dpi= 300)
# RBF核对于高度非线性的数据拟合很好。使用RBF估算SVM的重点在于误差惩罚参数C 以及控制敏感度的参数gamma 的确定

