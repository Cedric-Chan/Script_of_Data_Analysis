import sys
print(sys.path)                         # 使用virtualenv
quit()
deactivate                              # 退出virtualenv
python                                  # 命令行输入，进入Python环境

import statsmodels.api as sm            # 需要先import，避免其他包的打扰！
import pandas as pd
import scipy.misc
import scipy.special
from sklearn import model_selection   # 导入模型处理模块

Profit = pd.read_excel(r'C:\Users\123\Desktop\Predict to Profit.xlsx')                      # 导入数据
train, test = model_selection.train_test_split(Profit, test_size = 0.2, random_state=1234)  # 将数据集拆分为训练集和测试集

###模型的建立
# 根据train数据集建模（离散变量需加C(),即设置为哑变量）
model = sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + C(State)', data = train).fit() 
print('模型的偏回归系数分别为：\n', model.params)
# 删除test数据集中的Profit变量，用剩下的自变量进行预测（预测测试集的因变量，所以需先删去测试的因变量）
test_X = test.drop(labels = 'Profit', axis = 1)
pred = model.predict(exog = test_X)   # exog为用于测试集中其他自变量的值
print('对比预测值和实际值的差异：\n',pd.DataFrame({'Prediction':pred,'Real':test.Profit}))     # 同时显现真值与预测值
# 模型公式即为：Profit=58581.52+0.80RD_Spend-0.06Administration+0.01Marketing_Spend+927.39Florida-513.47New York


###显著性检验
##1 模型的显著性检验
import numpy as np
ybar = train.Profit.mean()            # 计算建模数据中，因变量的均值
p = model.df_model                    # p为统计变量个数!
n = train.shape[0]                    # n为统计观测个数!
model.fvalue                          # 返回模型中的F值(174.64)
# 对比理论F值
from scipy.stats import f
F_Theroy = f.ppf(q=0.95, dfn = p,dfd = n-p-1)   # 计算置信水平0.95下F分布（dfn,dfd）的理论值
print('F分布的理论值为: ',F_Theroy)    # 模型f值远大于理论F值，故拒绝原假设，即回归模型是显著的
##2 回归系数的显著性检验
model.summary()                       # 输出回归模型的所有指标
# 概率值p<0.05才有显著性，结果只有截距项和研发成本通过了回归系数的显著性检验


###回归模型的诊断（<异常值检验>，误差项独立，误差项正态，无多重共线性，线性相关性，方差齐性）
##1 正态性检验
# （一）定性的图形法——直方图
import scipy.stats as stats           # 导入第三方模块
# ！中文和负号的正常显示！
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 图中要素含因变量直方图，核密度曲线，理论正态分布
sns.distplot(a = Data.MOB, bins = 10, fit = stats.norm, norm_hist = True,
             hist_kws = {'color':'steelblue', 'edgecolor':'black'}, 
             kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'}, 
             fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
plt.legend()
plt.show()                            # 直观上可以认为利润变量服从正态分布
# （二）定性的图形法——PP图 & QQ图（残差的正态性检验）
pp_qq_plot = sm.ProbPlot(Data.MOB)
pp_qq_plot.ppplot(line = '45')
plt.title('P-P图')
pp_qq_plot.qqplot(line = 'q')
plt.title('Q-Q图')
plt.show()                            # 散点都比较均匀地落在直接附件，说明变量近似的服从正态分布
# （三）定量的非参数法——Shapiro检验（数据量小于5000使用Shapiro）
import scipy.stats as stats       
stats.shapiro(Data.MOB_B)          # 导入模块进行Shapiro检验（第二项p>0.05置信水平则接受利润因变量服从正态分布的原假设）
# （四）定量的非参数法——K-S检验（数据量大于5000使用）
# 生成正态分布和均匀分布随机数
rnorm = np.random.normal(loc = 5, scale=2, size = 10000)
runif = np.random.uniform(low = 1, high = 100, size = 10000)
# 对比检验
KS_Test1 = stats.kstest(rvs = rnorm, args = (rnorm.mean(), rnorm.std()), cdf = 'norm')
KS_Test2 = stats.kstest(rvs = runif, args = (runif.mean(), runif.std()), cdf = 'norm')
print(KS_Test1)                        # 将数据与随机生成的正态数据对比，p>0.05,即小概率错误未发生，接受正态分布的原假设
print(KS_Test2)                        # 将数据与随机生成的均匀分布数据对比，p<0.05,即小概率错误发生，拒绝均匀分布的原假设

##2 多重共线性检验
from statsmodels.stats.outliers_influence import variance_inflation_factor   # 导入方差膨胀因子VIF模块
# 自变量X(包含RD_Spend、Marketing_Spend和常数列1)
X = sm.add_constant(Profit.ix[:,['RD_Spend','Marketing_Spend']])             # 计算市场营销和研发成本之间的膨胀因子
vif = pd.DataFrame()                   # 构造空的数据框，用于存储VIF值
vif["features"] = X.columns      
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif                                    # 返回VIF值(对应的vif均低于10，即数据不存在多重共线性)

##3 线性相关性检验
# 第一步：Pearson相关系数法（只计算线性关系数值）
# 计算数据集Profit中每个自变量与因变量利润之间的Pearson相关系数，|ρ|>0.8为高度线性相关，|ρ|>0.5为中度线性相关，|ρ|<0.3为几乎不相关
Profit.drop('Profit', axis = 1).corrwith(Profit.Profit)      # 不相关只是不存在线性关系，还需制图观察散点的非线性关系
# 第二步：可视化观察（观察散点判断非线性关系并修正模型）
import matplotlib.pyplot as plt
import seaborn                         # seaborn的pairplot模块可绘制多个变量间的散点图矩阵
seaborn.pairplot(Profit.ix[:,['RD_Spend','Administration','Marketing_Spend','Profit']])     # (哑变量不在其中。效果Awesome!)
plt.show()                             # 根据先前分析和图例，只保留RD-Spend和Marketing-Spend两个自变量，对模型进行修正
# 模型修正
model0 = sm.formula.ols('Profit ~ RD_Spend + Marketing_Spend', data = train).fit()
model0.params                          # 修正模型的回归系数估计值（Profit=51902.11+0.79RD_Spend+0.02Marketing_Spend）

##4 异常值检验（多元线性回归模型容易受极端值影响）（必须先建立回归模型！）
# 方法：帽子矩阵，DFFIT准则，学生化残差，Cook距离
outliers = model.get_influence()
leverage = outliers.hat_matrix_diag    # 高杠杆值点（帽子矩阵）
dffits = outliers.dffits[0]            # dffits值
resid_stu = outliers.resid_studentized_external               # 学生化残差
cook = outliers.cooks_distance[0]      # cook距离
# 合并各种异常值检验的统计量值
contat1 = pd.concat([pd.Series(leverage, name = 'leverage'),pd.Series(dffits, name = 'dffits'),
                     pd.Series(resid_stu,name = 'resid_stu'),pd.Series(cook, name = 'cook')],axis = 1)
train.index = range(train.shape[0])    # 重设train数据的行索引（为了与这些求得的统计量合并）
profit_outliers = pd.concat([train,contat1], axis = 1)        # 将以上统计量与train数据集合并
profit_outliers.head()
# 选用标准化残差法判断异常点（标准化残差大于2即认为对应数据点为异常值）
outliers_ratio = sum(np.where((np.abs(profit_outliers.resid_stu)>2),1,0))/profit_outliers.shape[0]
outliers_ratio                         # 计算异常值数量的比例（小于5%可删，大于5%需要衍生哑变量）
none_outliers = profit_outliers.ix[np.abs(profit_outliers.resid_stu)<=2]   # 挑选出非异常的观测点重新建模
model1 = sm.formula.ols('Profit ~ RD_Spend + Marketing_Spend', data = none_outliers).fit()
model1.params                          # Profit=51827.42+0.80RD_Spend+0.02Marketing_Spend

##5 独立性检验（残差的独立性检验也是对因变量y的独立性检验）
model1.summary()                       # Durbin-Watson值在2左右即表明残差项之间不相关

##6 方差齐性检验（方差不随自变量的变动呈现某种趋势）
# <1>图形法（散点图）
ax1 = plt.subplot2grid(shape = (2,1), loc = (0,0))            # 设置第一张子图的位置
ax1.scatter(none_outliers.RD_Spend, (model1.resid-model1.resid.mean())/model1.resid.std())  # 绘制散点图
# 给第一张子图添加水平参考线,添加x轴和y轴标签
ax1.hlines(y = 0 ,xmin = none_outliers.RD_Spend.min(),xmax = none_outliers.RD_Spend.max(), color = 'red', linestyles = '--')
ax1.set_xlabel('RD_Spend')
ax1.set_ylabel('Std_Residual')
ax2 = plt.subplot2grid(shape = (2,1), loc = (1,0))            # 设置第二张子图的位置
ax2.scatter(none_outliers.Marketing_Spend, (model1.resid-model1.resid.mean())/model1.resid.std())
# 给第二张子图添加水平参考线,添加x轴和y轴标签
ax2.hlines(y = 0 ,xmin = none_outliers.Marketing_Spend.min(),xmax = none_outliers.Marketing_Spend.max(), color = 'red', linestyles = '--')
ax2.set_xlabel('Marketing_Spend')
ax2.set_ylabel('Std_Residual')
# 调整子图之间的水平间距和高度间距
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()                             # 标准化残差并未随自变量的变动而呈喇叭形，即符合方差齐性的前提假设
# <2>BP检验（统计检验法）
sm.stats.diagnostic.het_breushpagan(model1.resid, exog_het = model1.model.exog)             # 通过statsmodels模块的het_breushpagan函数实现
# （第二项大于0.05则接受残差方差为常数的原假设；第四项为F统计量p值，大于0.05进一步证明残差平方项与自变量之间独立，进而说明残差方差齐性）


###回归模型的预测（已确立了合理有效的最终模型model1）
# model1对测试集的预测
pred1 = model1.predict(exog = test.ix[:,['RD_Spend','Marketing_Spend']])
plt.scatter(x = test.Profit, y = pred1)# 绘制预测值与实际值的散点图
# 添加斜率为1，截距项为0的参考线
plt.plot([test.Profit.min(),test.Profit.max()],[test.Profit.min(),test.Profit.max()],
        color = 'red', linestyle = '--')
plt.xlabel('实际值')                    # 添加轴标签
plt.ylabel('预测值')
plt.show()























