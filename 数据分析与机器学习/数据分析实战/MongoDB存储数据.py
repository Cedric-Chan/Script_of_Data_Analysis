import pandas as pd
import pymongo

r_filenameCSV = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter01/realEstate_trans.csv'
csv_read = pd.read_csv(r_filenameCSV)

csv_read['sale_date'] = pd.to_datetime(csv_read['sale_date'])

# 连接到数据库
client = pymongo.MongoClient()
# 将数据库命名为packt并存储于db
db = client['packt']
# 连接到real_estate集合
real_estate = db['real_estate']

# 在插入数据前清空集合中已有文档
if real_estate.count() > 0:
    real_estate.remove()

# 插入数据
real_estate.insert(csv_read.to_dict(orient='records'))

# 输出邮编为xx的头十行记录
sales = real_estate.find({'zip': {'$in': [95841, 95842]}})  # $有指定的作用
for sale in sales.sort('_id').limit(10):
    print(sale)