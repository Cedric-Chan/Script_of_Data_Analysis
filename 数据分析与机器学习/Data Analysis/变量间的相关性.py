import pandas as pd 

# 读取的文件位置
r_filenameCSV = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter02/' + 'realEstate_trans_full.csv'

# 写入的文件位置
w_filenameCSV = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter02/' + 'realEstate_corellations.csv'

# 只读取部分变量
csv_read = pd.read_csv(r_filenameCSV)
csv_read = csv_read[['beds','baths','sq__ft','price']]


# 遍历计算相关系数
coefficients = ['pearson', 'kendall', 'spearman'] # 皮尔逊，肯达尔，斯皮尔曼，后两个对于非正态分布的随机变量不敏感

csv_corr = {}
for coefficient in coefficients:
    csv_corr[coefficient] = csv_read.corr(method=coefficient).transpose()

# 存入文件
with open(w_filenameCSV,'w') as write_csv:
    for corr in csv_corr:
        write_csv.write(corr + '\n')
        write_csv.write(csv_corr[corr].to_csv(sep=','))
        write_csv.write('\n')