import requests, re, time, csv
import math
from bs4 import BeautifulSoup as BS
from selenium import webdriver
import pandas as pd #用来操作csv的库，这次用来创建表格，存储数据
import numpy as np
import codecs
import urllib
from fake_useragent import UserAgent  # 用于生成随机请求头

data = pd.read_excel('desktop/新冠病毒科普_合并筛选6.27.xlsx')

#? BV号转AV号
def BvToAv(Bv):
    # 1.去除Bv号前的"Bv"字符
    BvNo1 = Bv[2:]
    keys = {
        '1':'13', '2':'12', '3':'46', '4':'31', '5':'43', '6':'18', '7':'40', '8':'28', '9':'5',
        'A':'54', 'B':'20', 'C':'15', 'D':'8', 'E':'39', 'F':'57', 'G':'45', 'H':'36', 'J':'38', 'K':'51', 'L':'42', 'M':'49', 'N':'52', 'P':'53', 'Q':'7', 'R':'4', 'S':'9', 'T':'50', 'U':'10', 'V':'44', 'W':'34', 'X':'6', 'Y':'25', 'Z':'1',
        'a': '26', 'b': '29', 'c': '56', 'd': '3', 'e': '24', 'f': '0', 'g': '47', 'h': '27', 'i': '22', 'j': '41', 'k': '16', 'm': '11', 'n': '37', 'o': '2',
        'p': '35', 'q': '21', 'r': '17', 's': '33', 't': '30', 'u': '48', 'v': '23', 'w': '55', 'x': '32', 'y': '14','z':'19'
    }
    # 2. 将key对应的value存入一个列表
    BvNo2 = []
    for index, ch in enumerate(BvNo1):
        BvNo2.append(int(str(keys[ch])))

    # 3. 对列表中不同位置的数进行*58的x次方的操作
    BvNo2[0] = int(BvNo2[0] * math.pow(58, 6));
    BvNo2[1] = int(BvNo2[1] * math.pow(58, 2));
    BvNo2[2] = int(BvNo2[2] * math.pow(58, 4));
    BvNo2[3] = int(BvNo2[3] * math.pow(58, 8));
    BvNo2[4] = int(BvNo2[4] * math.pow(58, 5));
    BvNo2[5] = int(BvNo2[5] * math.pow(58, 9));
    BvNo2[6] = int(BvNo2[6] * math.pow(58, 3));
    BvNo2[7] = int(BvNo2[7] * math.pow(58, 7));
    BvNo2[8] = int(BvNo2[8] * math.pow(58, 1));
    BvNo2[9] = int(BvNo2[9] * math.pow(58, 0));

    # 4.求出这10个数的合
    sum = 0
    for i in BvNo2:
        sum += i
    # 5. 将和减去100618342136696320
    sum -= 100618342136696320
    # 6. 将sum 与177451812进行异或
    temp = 177451812
    return sum ^ temp

#* test
bv = data['BV']
BvToAv(bv[0])
BvToAv('BV1M7411G775')

#* 应用在全部bv
av=bv.apply(BvToAv)
av

av_bv = pd.DataFrame(av)
av_bv['bv']=bv
av_bv.columns=['AV','BV']
av_bv.to_csv('desktop/av_bv.csv')

#? BV号转AV号
import requests
import re
import os
import sys
import json

# 视频AV号列表
aid_list = av

# 获取一个AV号视频下所有评论
def getAllCommentList(item):
    info_list = []
    url = "http://api.bilibili.com/x/reply?type=1&oid=" + str(item) + "&pn=1&nohot=1&sort=0"
    r = requests.get(url)
    numtext = r.text
    json_text = json.loads(numtext)
    commentsNum = json_text["data"]["page"]["count"]
    page = 10
    try:
        for n in range(1,page):
            url = "https://api.bilibili.com/x/v2/reply?jsonp=jsonp&pn="+str(n)+"&type=1&oid="+str(item)+"&sort=1&nohot=1"
            req = requests.get(url)
            text = req.text
            json_text_list = json.loads(text)
            for i in json_text_list["data"]["replies"]:
                info_list.append(i["content"]["message"])
            
            info_list=pd.DataFrame(info_list)
            info_list['No_AV']=str(item)
    except:
        pass
    filename = 'C:/Users/60448/Desktop/comments/'+ str(item) + ".csv"
    info_list.to_csv(filename,encoding='utf_8_sig',header=None,index=False)
    return item

getAllCommentList(aid_list[0])

#todo 正式开爬
for i in aid_list:
    try:
        getAllCommentList(i)
    except:
        pass



#todo 合并多个CSV
import os
import pandas as pd

path = 'C:/Users/60448/Desktop/comments/'   #设置csv所在文件夹
files = os.listdir(path)  #获取文件夹下所有文件名

csv_list = []
for x in files:
    csv_list.append(path+x)

outputfile='desktop/comments.csv'

#合并csv
for i in csv_list:
    fr = open(i,'rb').read()   #读出来
    with open(outputfile,'ab') as f:
        f.write(fr)   #上下文管理器写进去
