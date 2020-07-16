http://comment.bilibili.com/{cid}.xml  查看该视频弹幕列表
# 有了弹幕数据后，我们需要先将解析好，并保存在本地，方便进一步的加工处理，如制成词云图进行展示

import requests
from bs4 import BeautifulSoup
import pandas as pd

# 请求弹幕数据
url = 'http://comment.bilibili.com/97160161.xml'          
html = requests.get(url).content
# 解析弹幕数据
html_data = str(html, 'utf-8')
bs4 = BeautifulSoup(html_data, 'lxml')    # 使用lxml解析器
results = bs4.find_all('d')               # 弹幕都存放在<d>……</d>框架内
comments = [comment.text for comment in results]
comments_dict = {'comments': comments}
# 将弹幕数据保存在本地
bili = pd.DataFrame(comments_dict)
bili.to_csv('desktop/bilibili.csv')

from wordcloud import WordCloud, ImageColorGenerator
import PIL.Image as Image
import matplotlib.pyplot as plt
import pandas as pd
import jieba
import re
import numpy as np

'''按行截取不中间断开
wd=pd.read_excel(r'desktop/hospital.xlsx')
name=[i[0] for i in wd[['Illness']].values]          # 双中括号保证词语完整
value=[int(i[0]) for i in wd[['Value']].values]

word=zip(name,value)                                # 两个列表对应合并为元组列表！！
'''

# 进行分词，并用空格连起来
text_from_file=open('desktop/bilibili.csv','r',encoding='UTF-8').read()   #  encoding='UTF-8'解决不能读取中文csv
Word_spilt_jieba = jieba.cut(text_from_file,cut_all = False)
word_space_split = ' '.join(Word_spilt_jieba)


coloring = np.array(Image.open('desktop/junhui.jpg'))     # 注意斜杠方向
font=r'C:\Windows\Fonts\BDZYJT.ttf'                       # 中文字体路径
my_wordcloud = WordCloud(background_color='white', max_words=2000,
                         mask=coloring, max_font_size=80, random_state=30,width=1400, height=1400, font_path=font,margin=2).generate(word_space_split)
image_colors = ImageColorGenerator(coloring)
# plt.imshow(my_wordcloud.recolor(color_func=image_colors))  红色星
plt.rcParams['font.sans-serif'] = ['SimHei']              # 两行解决乱码！！！
plt.rcParams['axes.unicode_minus'] = False                # 两行解决乱码！！！
plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()

