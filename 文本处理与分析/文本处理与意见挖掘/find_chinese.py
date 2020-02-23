# -*- coding: utf-8 -*-

import re

def find_unchinese(file):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    unchinese = re.sub(pattern, '', file)
    return unchinese
 
def find_chinese(file):
    pattern = re.compile(u'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern,"",file)
    return chinese

def delete_re(file):
    char_list = list(file) #把字符串转化列表自动按单个字符分词了
    #print(char_list)

    list1 = []
    list1.append(char_list[0])
    list2 = ['']

    #记录要删除的索引
    del1 = []
    i = 0
    while (i<len(char_list)):
        i = i+1
        #这里是对后面没有词汇的时候对列表1和列表2判断一次重复
        if i == len(char_list):
            if list1 == list2:
                m = len(list2)
                for x in range(i-m,i):
                    del1.append(x)
        else:
            
            if char_list[i] == list1[0] and list2==['']:
                #print('词汇和list1相同，list2为空，将词加入list2')
                list2[0]=char_list[i] #这里初始化用append会让lisr2初始化为['','**']
            elif char_list[i] != list1[0] and list2==['']:
                #print('词汇和1不同，2为空，将词加入1')
                list1.append(char_list[i])

            #触发判断
            elif char_list[i] != list1[0] and list2 !=['']:
                if list1 == list2 and len(list2)>=2:
                    #print('词和1不同，2不为空，判断1和2重复')
                    m = len(list2)
                    #删除列表2里的内容，列表1本来的内容不用再去判断重复了
                    for x in range(i-m,i):
                        del1.append(x)
                    list1= ['']
                    list2 = ['']
                    list1[0]=char_list[i]
                else:
                    #print('词和1不同，2不为空，判断1和2不重复')
                    list2.append(char_list[i])

            #触发判断
            elif char_list[i] == list1[0] and list2 != ['']:
                if list1 == list2:
                    #print('词和1相同，2不为空，判断1和2重复')
                    m = len(list2)
                    #删除列表2里的内容，列表1需要再去和后面的词汇继续判断重复
                    for x in range(i-m,i):
                        del1.append(x)

                    list2 = ['']
                    list2[0]=char_list[i]
                else:
                    #print('词和1相同，2不为空，判断1和2不重复')
                    #逻辑对书本上进行了修改，书上是清空列表1和2，就是保留现在列表1和2内容不做删除，这里只保留1，列表2内容还需要做对比
                    list1 = list2
                    list2 = ['']
                    list2[0]=char_list[i]


    a = sorted(del1) #从数字更大的索引删起，这样就不用考虑元素删除后索引的变化问题
    t = len(a) - 1
    while(t>=0):
        del char_list[a[t]]
        t = t-1
    str1 = ''.join(char_list)
    str2 = str1.strip()
    return str2


def delete_reback(file):
    char_list = list(file) #把字符串转化列表自动按单个字符分词了
    char_list = char_list[::-1]
    #print(char_list)

    list1 = []
    list1.append(char_list[0])
    list2 = ['']

    #记录要删除的索引
    del1 = []
    i = 0
    while (i<len(char_list)):
        i = i+1
        #这里是对后面没有词汇的时候对列表1和列表2判断一次重复
        if i == len(char_list):
            if list1 == list2:
                m = len(list2)
                for x in range(i-m,i):
                    del1.append(x)
        else:
            
            if char_list[i] == list1[0] and list2==['']:
                #print('词汇和list1相同，list2为空，将词加入list2')
                list2[0]=char_list[i] #这里初始化用append会让lisr2初始化为['','**']
            elif char_list[i] != list1[0] and list2==['']:
                #print('词汇和1不同，2为空，将词加入1')
                list1.append(char_list[i])

            #触发判断
            elif char_list[i] != list1[0] and list2 !=['']:
                if list1 == list2 and len(list2)>=2:
                    #print('词和1不同，2不为空，判断1和2重复')
                    m = len(list2)
                    #删除列表2里的内容，列表1本来的内容不用再去判断重复了
                    for x in range(i-m,i):
                        del1.append(x)
                    list1= ['']
                    list2 = ['']
                    list1[0]=char_list[i]
                else:
                    #print('词和1不同，2不为空，判断1和2不重复')
                    list2.append(char_list[i])

            #触发判断
            elif char_list[i] == list1[0] and list2 != ['']:
                if list1 == list2:
                    #print('词和1相同，2不为空，判断1和2重复')
                    m = len(list2)
                    #删除列表2里的内容，列表1需要再去和后面的词汇继续判断重复
                    for x in range(i-m,i):
                        del1.append(x)

                    list2 = ['']
                    list2[0]=char_list[i]
                else:
                   # print('词和1相同，2不为空，判断1和2不重复')
                    #逻辑对书本上进行了修改，书上是清空列表1和2，就是保留现在列表1和2内容不做删除，这里只保留1，列表2内容还需要做对比
                    list1 = list2
                    list2 = ['']
                    list2[0]=char_list[i]


    a = sorted(del1) #从数字更大的索引删起，这样就不用考虑元素删除后索引的变化问题
    t = len(a) - 1
    while(t>=0):
        del char_list[a[t]]
        t = t-1
    str1 = ''.join(char_list)
    str2 = str1.strip()
    str3 = str2[::-1]
    return str3


def delete_copy(file):
    j=0
    str2=file
    while(j<len(str2)):
        char_list = list(str2) #把字符串转化列表自动按单个字符分词了
        list1 = []
        list1.append(char_list[j])
        j+=1
        list2 = ['']
    
        #记录要删除的索引
        del1 = []
        i = 0
        while (i<len(char_list)):
            i = i+1
            #这里是对后面没有词汇的时候对列表1和列表2判断一次重复
            if i == len(char_list):
                if list1 == list2:
                    m = len(list2)
                    for x in range(i-m,i):
                        del1.append(x)
            else:
                
                if char_list[i] == list1[0] and list2==['']:
                    #print('词汇和list1相同，list2为空，将词加入list2')
                    list2[0]=char_list[i] #这里初始化用append会让lisr2初始化为['','**']
                elif char_list[i] != list1[0] and list2==['']:
                    #print('词汇和1不同，2为空，将词加入1')
                    list1.append(char_list[i])
    
                #触发判断
                elif char_list[i] != list1[0] and list2 !=['']:
                    if list1 == list2 and len(list2)>=2:
                        #print('词和1不同，2不为空，判断1和2重复')
                        m = len(list2)
                        #删除列表2里的内容，列表1本来的内容不用再去判断重复了
                        for x in range(i-m,i):
                            del1.append(x)
                        list1= ['']
                        list2 = ['']
                        list1[0]=char_list[i]
                    else:
                        #print('词和1不同，2不为空，判断1和2不重复')
                        list2.append(char_list[i])
    
                #触发判断
                elif char_list[i] == list1[0] and list2 != ['']:
                    if list1 == list2:
                        #print('词和1相同，2不为空，判断1和2重复')
                        m = len(list2)
                        #删除列表2里的内容，列表1需要再去和后面的词汇继续判断重复
                        for x in range(i-m,i):
                            del1.append(x)
    
                        list2 = ['']
                        list2[0]=char_list[i]
                    else:
                        #print('词和1相同，2不为空，判断1和2不重复')
                        #逻辑对书本上进行了修改，书上是清空列表1和2，就是保留现在列表1和2内容不做删除，这里只保留1，列表2内容还需要做对比
                        list1 = list2
                        list2 = ['']
                        list2[0]=char_list[i]
        a = sorted(del1) #从数字更大的索引删起，这样就不用考虑元素删除后索引的变化问题
        t = len(a) - 1
        while(t>=0):
            del char_list[a[t]]
            t = t-1
        str1 = ''.join(char_list)
        str2 = str1.strip()
    return str2



