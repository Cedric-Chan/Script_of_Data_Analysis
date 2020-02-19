# 从MongoDB读取数据，用python取样

import pymongo
import pandas as pd
import numpy as np

# 确定抽样比例
strata_frac = 0.2
# 输出环境
w_filenameSample = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter02/realEstate_sample.csv'

# 连接到MongoDB
client = pymongo.MongoClient()
db = client['packt']
real_estate = db['real_estate']

beds = [2,3,4]
sales = pd.DataFrame.from_dict(
    list(
        real_estate.find(
            {
                'beds': {'$in': beds} # 指定筛选条件
            }, 
            {
                '_id': 0, # 用0,1表示是否加入字段
                'zip': 1, 
                'city': 1, 
                'price': 1,
                'beds': 1,
                'sq__ft': 1
            }
        )
    )
)

# 取样
sample = pd.DataFrame()
for bed in beds:
    sample = sample.append(
        sales[sales.beds == bed].sample(frac=strata_frac),
        ignore_index=True  # 忽略附加DataFrame的索引值
    )

# 抽样数目
strata_sampled_counts = sample['beds'].value_counts()
print('Sampled: ', strata_sampled_counts)

# 写入文件并存储
with open(w_filenameSample,'w') as write_csv:
    write_csv.write(sample.to_csv(sep=',', index=False))