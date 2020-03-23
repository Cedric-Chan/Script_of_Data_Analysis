'''
5 个优雅的 Python Numpy 函数，有助于高效、简洁的数据处理
'''
import numpy as np

## 1 在 reshape 函数中使用参数-1
'''
可以将新形状中的一个参数赋值为-1。这仅仅表明它是一个未知的维度
'''
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

a.reshape(4,-1)  # Numpy 会自动算出这个未知的维度
a.reshape(3,-1)   # 如果我们尝试 reshape 不兼容的形状或者是给定的未知维度参数多于 1 个，那么将会报错


## 2 Argpartition：在数组中找到最大的 N 个元素
'''
Numpy 的 argpartion 函数可以高效地找到 N 个最大值的索引并返回 N 个值。在给出索引后，我们可以根据需要进行值排序
'''
array = np.array([10, 7, 4, 3, 2, 2, 5, 9, 0, 4, 6, 0])
index = np.argpartition(array, -5)[-5:]   # 按顺序给出最大的五个值的索引

np.sort(array[index])   # 按索引返回最大的五个值


## 3 Clip：使数组中的值保持在一定区间内
'''
Numpy clip () 函数用于对数组中的值进行限制。给定一个区间范围，区间范围外的值将被截断到区间的边界上。例如，如果指定的区间是 [-1,1]，小于-1 的值将变为-1，而大于 1 的值将变为 1。
'''
array = np.array([10, 7, 4, 3, 2, 2, 5, 9, 0, 4, 6, 0])
np.clip(array,2,6)   # 上下限设置为2-6


## 4 Extract：从数组中提取符合条件的元素
arr = np.arange(10)
condition = np.mod(arr, 3)==0  # mod为整除条件

np.extract(condition, arr)  # 提取数组中满足整除3的条件的元素

np.extract(((arr > 2) & (arr < 8)), arr)  # 提取 2 < x < 8 的元素
np.extract(((arr > 2) | (arr < 8)), arr)  # 提取>2或者<8的x元素


## 5 setdiff1d：找到仅在 A 数组中有而 B 数组没有的元素
'''
返回数组中不在另一个数组中的独有元素。这等价于两个数组元素集合的差集
'''
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([3,4,7,6,7,8,11,12,14])
c = np.setdiff1d(a,b)
c