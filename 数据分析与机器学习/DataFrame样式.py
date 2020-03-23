import pandas as pd
import numpy as np

'''
设置 DataFrame 不同列的样式，如千分号、小数位数、货币符号、日期格式，百分比等；

突出显示 DataFrame 中某一列里符合某些条件的数据，如，按不同颜色显示某列中的最大值与最小值；

以不同渐变色展示每一行数据占总数据的比例大小，占比越大，颜色越深，占比越小，则颜色越浅；

实现类似 Excel 迷你条形图的功能，根据数据大小，在单元格里显示迷你条形图；

利用 Sparklines 支持库，配合 Pandas 绘制迷你走势图
'''
df=pd.read_clipboard()
#———————————————————————————————————————————————————————————————————————————— example 1 ————————————————————————————————————————————————————————————————

# 创建样式字符字典
format_dict={'Date':'{:%y-%m-%d}','Close':'${:.2f}','Volume':'{:,}','Rate':'{:.2%}'}  # 日期是月-日-年的格式，闭市价有美元符，交易量有千分数符，交易率是带两位小数的百分比形式
data.style.format(format_dict).hide_index()

# 隐藏索引，闭市价最小值用红色显示，最大值用浅绿色显示
(stocks.style.format(format_dict)
.hide_index()  # 隐藏索引
.highlight_min('Close',color='red')
.highlight_max('Close',color='lightgreen')
)


# 背景色渐变的样式
(stocks.style.format(format_dict)
.hide_index()  # 隐藏索引
.background_gradient(subset='Volume',cmap='Blues')  # 对于交易量背景色按照蓝色渐变色布景  （颜色可选"BuGn"）
)


# 迷你条形图样式
(stocks.style.format(format_dict)
.hide_index()  # 隐藏索引
.bar(color='lightblue', vmin=100_000, subset=['Volume'], align='zero')  # 对于交易量背景色按照蓝色迷你条形图布景,vmin 是基准值,subset 是针对的列，align 则代表对齐方式
.bar(color='#FFA07A', vmin=0, subset=['Rate'], align='zero')
.set_caption('2019年11月20日股票价格')  # 添加标题
)


# Sparklines - 走势图
import sparklines

def sparkline_str(x):         # 定义调用 sparklines 的函数
    bins=np.histogram(x)[0]
    sl = ' '.join(sparklines(bins))    
    return sl

sparkline_str.__name__ = "走势图"

df.groupby('姓名')['数量', '金额'].agg(['mean', sparkline_str])   # 在 groupby 函数里调用定义的 sparkline_str 函数

#———————————————————————————————————————————————————————————————————————————— example 2 ————————————————————————————————————————————————————————————————

monthly_sales = df.groupby([pd.Grouper(key='日期', freq='M')])['金额'].agg(['sum']).reset_index()  # 按销售日期计算每月销售金额
monthly_sales['月占比'] = monthly_sales['sum']/df['金额'].sum()  # 计算月销售占年度销售总额的比例

(
    df.groupby('姓名')['金额']
    .agg(['mean', 'sum'])
    .style.format('${0:,.2f}')  # 前面加$符号，0到所有行，.2f代表两位小数
)

