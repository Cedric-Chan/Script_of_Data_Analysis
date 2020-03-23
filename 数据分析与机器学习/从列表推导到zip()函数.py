import numpy as np

#——————————————————————————————————————————————————————————————————————字符串运算————————————————————————————————————————————————————————————————————
'''
字符串本质上也是一种元组，但是字符串有很多「运算」方式。最直观的是字符串的 + 和 * 运算，它们分别表示重复和连接
'''
my_string = "Hello World!"

print(my_string * 2)  # *表示连续两次
print(my_string + " I love Python" * 2)  # +表示连接

my_list = [1,2,3,4,5]
print(my_list[::-1])  # 使用 [::-1] 进行反向索引

# 如果列表元素都是字符串，那么我们可以快速地使用 join() 方法将所有元素拼接在一起
word_list = ["awesome", "is", "this"]
print(' '.join(word_list[::-1]) + '!')   # 如果将句子拆分为了字符，那么处理后的合并就需要使用 join() 了


#——————————————————————————————————————————————————————————————————————列表推导式————————————————————————————————————————————————————————————————————
'''
[ expression for item in list if conditional ].
列表推导式真的非常强大，它不仅在速度上比一般的方法快，同时直观性、可读性都非常强
'''
my_list = [1, 2, 3, 4, 5]

def stupid_func(x):
    return x**2 +5

print([stupid_func(x) for x in my_list if x % 2 != 0])


#—————————————————————————————————————————————————————————————————————Lambda 和 Map————————————————————————————————————————————————————————————————————
'''
Lambda 最常执行一些直观的运算，它并不需要标准的函数定义，而且也不需要新的函数名再次调用。因为当我们想执行一些简单运算时，可以不需要定义真实函数就能完成。

'''
stupid_func=(lambda x: x**2 + 5)  # 将上各个def函数简写为lamba函数
print(stupid_func(5))

my_list=list(range(-2,3))
sorted(my_list, key = lambda x : x ** 2)  # 利用lambda函数实现更自由的排序

'''
Map 是一个简单的函数，它可以将某个函数应用到其它一些序列元素，例如列表。
'''
list(map(lambda x, y : x * y, [1, 2, 3], [4, 5, 6]))


#—————————————————————————————————————————————————————————————————————单行条件语句————————————————————————————————————————————————————————————————————
x = int(input())
print("Excellent" if x >= 90 else "Great" if 80 <= x < 90 else "Good" if 60<=x<80 else "Failed")   # 最后一个else语句没有if


#—————————————————————————————————————————————————————————————————————————zip()————————————————————————————————————————————————————————————————————
'''
使用 zip() 函数，我们可以将两个列表拼接在一起
zip 将两个等长的列表变为了一对一对的，即列表内配对的元组
'''
first_names = ["Peter", "Christian", "Klaus"]
last_names = ["Jensen", "Smith", "Nistrup"]
print([' '.join(x) for x in zip(first_names, last_names)]) # ['Peter Jensen', 'Christian Smith', 'Klaus Nistrup']


# *zip的功能则是解包
array = [[ 'a' ,  'b' ], [ 'c' ,  'd' ], [ 'e' ,  'f' ]]
transposed = zip(*array)
print(transposed)  # [( a ,  c ,  e ), ( b ,  d ,  f )]

# ? 将两个列表合并为字典

def to_dictionary(keys, values):
    return dict(zip(keys, values))


keys = ["a", "b", "c"]
values = [2, 3, 4]
print(to_dictionary(keys, values))   # { a : 2,  c : 4,  b : 3}


