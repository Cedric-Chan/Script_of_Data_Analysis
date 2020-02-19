###岭回归（可解决线性回归模型中系数矩阵不可逆的问题，但无法降低模型的复杂度，始终保留建模的所有变量）
# 岭回归的关键是找到一个合理的λ值来平衡模型的方差和偏差，进而得到更符合实际的岭回归系数（岭回归模型的系数是关于λ的函数）
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge,RidgeCV      # 导入sklearn 中的ridge类
import matplotlib.pyplot as plt
# 读取糖尿病数据集
diabetes = pd.read_excel(r'C:\Users\123\Desktop\diabetes.xlsx', sep = '')
# 构造自变量（剔除患者性别、年龄和因变量）
predictors = diabetes.columns[2:-1]
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(diabetes[predictors], diabetes['Y'], 
                                                                    test_size = 0.2, random_state = 1234 )
# 第一步：粗略确定λ值的可视化方法（对比不同λ值和对应的回归系数的折线图，粗略确定合理的λ）
Lambdas = np.logspace(-5, 2, 200)	                # 构造不同的Lambda值	
ridge_coefficients = []                             # 构造空列表，用于存储模型的偏回归系数
for Lambda in Lambdas:                              # 循环迭代不同的Lambda值
    ridge = Ridge(alpha = Lambda, normalize=True)   # alpha项需赋值lambda的参数，对数据集做标准化处理
    ridge.fit(X_train, y_train)
    ridge_coefficients.append(ridge.coef_)
# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
plt.plot(Lambdas, ridge_coefficients)               # 绘制Lambda与回归系数的关系
plt.xscale('log')                                   # 对x轴作对数变换(lambda的范围已设为logspace)
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.show()
# 第二步：岭回归模型的交叉验证（精确计算λ值）
# 设置交叉验证的参数，对数据集做标准化处理，选择均方差作为评估模型的度量方法，对于每一个λ值都执行10重交叉验证
ridge_cv = RidgeCV(alphas = Lambdas, normalize=True, scoring='neg_mean_squared_error', cv = 10)
ridge_cv.fit(X_train, y_train)                      # 模型拟合     
ridge_best_Lambda = ridge_cv.alpha_                 # 返回最佳的lambda值
ridge_best_Lambda
# 第三步：构建模型（基于最佳λ）
ridge = Ridge(alpha = ridge_best_Lambda, normalize=True)                      # 基于最佳的Lambda值建模
ridge.fit(X_train, y_train)
# 返回岭回归系数
pd.Series(index = ['Intercept'] + X_train.columns.tolist(),data = [ridge.intercept_] + ridge.coef_.tolist())
# (Y=-324.75+6.21BMI+0.93BP-0.50S1+0.22S2+0.04S3+4.25S4+52.1S5+0.38S6)
# 第四步：基于模型对测试集数据进行预测对比
from sklearn.metrics import mean_squared_error
ridge_predict = ridge.predict(X_test)
RMSE = np.sqrt(mean_squared_error(y_test,ridge_predict))                      # 预测效果验证（使用均方根误差RMSE）
RMSE                                                # RMSE值越小则拟合越好 


###LASSO回归可在缩减回归系数的过程中将不重要的回归系数缩减为0，达到变量筛选的功能
from sklearn.linear_model import Lasso,LassoCV      # 导入sklearn中的lasso类
lasso_coefficients = []                             # 构造空列表，用于存储模型的偏回归系数
for Lambda in Lambdas:                              # lambda在岭回归中定义过了
    lasso = Lasso(alpha = Lambda, normalize=True, max_iter=10000)
    lasso.fit(X_train, y_train)                     # 数据集在岭回归中已拆分过
    lasso_coefficients.append(lasso.coef_)
# 绘制Lambda与回归系数的关系,概略确定λ值
plt.plot(Lambdas, lasso_coefficients)
plt.xscale('log')                                   # 对x轴作对数变换
plt.xlabel('Lambda')
plt.ylabel('Cofficients')
plt.show()
# LASSO回归模型的交叉验证
lasso_cv = LassoCV(alphas = Lambdas, normalize=True, cv = 10, max_iter=10000)
lasso_cv.fit(X_train, y_train)
lasso_best_alpha = lasso_cv.alpha_                  # 输出最佳的lambda值
lasso_best_alpha
# 基于最佳的lambda值建模
lasso = Lasso(alpha = lasso_best_alpha, normalize=True, max_iter=10000)
lasso.fit(X_train, y_train)
# 返回LASSO回归的系数
pd.Series(index = ['Intercept'] + X_train.columns.tolist(),data = [lasso.intercept_] + lasso.coef_.tolist())
# （Y=-278.56+6.19BMI+0.86BP-0.13S1-0.49S3+44.49S5+0.32S6）
lasso_predict = lasso.predict(X_test)               # 基于模型对测试集数据进行预测
RMSE = np.sqrt(mean_squared_error(y_test,lasso_predict))
RMSE                                                # 返回均方根误差RMSE（相比岭回归下降了0.8，提升了拟合效果）

###加入多元回归模型作为参照对比
from statsmodels import api as sms
# 为自变量X添加常数列1，用于拟合截距项
X_train2 = sms.add_constant(X_train)
X_test2 = sms.add_constant(X_test)
linear = sms.formula.OLS(y_train, X_train2).fit()   # 构建多元线性回归模型
linear.params
linear_predict = linear.predict(X_test2)            # 模型的预测
RMSE = np.sqrt(mean_squared_error(y_test,linear_predict))
RMSE                                                # 返回均方根误差RMSE（三个模型中均方根误差最大）

