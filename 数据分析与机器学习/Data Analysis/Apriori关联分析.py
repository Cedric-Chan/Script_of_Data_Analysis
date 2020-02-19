''' 关联规则的核心就是找到事物发展的前后关联性，研究序列关联可以来推测事物未来的发展情况，并根据预测的发展情况进行事物的分配和安排
    支持度（Support）揭示了A=a和B=b同时出现的概率，
    置信度（Confidence）揭示了当A=a出现时，B=b是否会一定出现的概率
    提升度（Lift）表示含有X的条件下同时含有Y的概率，与不含X的条件下却含Y的概率之比。Lift大于1即为关联，大于指定正整数是强关联'''

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel('./Online Retail.xlsx')
df.head()

# 一、数据预处理，选定样本
df['Description'] = df['Description'].str.strip()        # 去除首尾空格
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)    # 删除发票ID"InvoiceNo"为空的数据记录
df['InvoiceNo'] = df['InvoiceNo'].astype('str')          # 发票ID"InvoiceNo"字段转为字符型
df = df[~df['InvoiceNo'].str.contains('C')]              # 删除发票ID"InvoiceNo"不包含“C”的记录

# 二、构建用于关联的数据集
# 方法1：使用pivot_table函数
basket = df[df['Country'] =="France"].pivot_table(columns = "Description",index="InvoiceNo",
              values="Quantity",aggfunc=np.sum).fillna(0)# fillna(0)将空值转为0
basket.head(20)

# 方法2：groupby + unstack
basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# 三、将数值型字段转为0/1变量
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)              # applymap()将encode_units在数据集每个单元格执行
basket_sets.drop('POSTAGE', inplace=True, axis=1)        # 删除购物篮中的邮费项（POSTAGE）

# 四、使用算法包进行关联规则运算
frequent_itemsets = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
# antecedants前项集，consequents后项集，support支持度，confidence置信度，lift提升度

# 五、结果检视
rules[ (rules['lift'] >= 5) & (rules['confidence'] >= 0.8) ].sort_values("lift",ascending = False)
# 选取置信度（confidence）大于0.8且提升度（lift）大于5的规则，按lift降序排序