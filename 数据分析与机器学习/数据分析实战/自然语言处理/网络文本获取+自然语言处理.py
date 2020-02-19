import urllib.request as u
import bs4 as bs

#——————————————————————————————————————————————————————————————————从网页提取文本————————————————————————————————————————————————————————————————————

# 报纸文章链接
st_url = 'http://www.seattletimes.com/nation-world/obama-starts-2016-with-a-fight-over-gun-control/'

# 读取网页内容
with u.urlopen(st_url) as response:
    html = response.read()

# 使用beautifulsoup解析HTML
read = bs.BeautifulSoup(html, 'lxml')

# 找到文章标签
article = read.find('article')

# 找到文章所有段落
all_ps = article.find_all('p')  # 文本都在段落中

# 存放文章主体的对象
body_of_text = []

# 获取标题
body_of_text.append(read.title.get_text())  # .get_text()用于剥落形如<title>、</title>标签，只提取标签对之间的内容
print(read.title)

# 将段落放入列表
for p in all_ps:
    body_of_text.append(p.get_text())

# 不需要页面底部无用信息
body_of_text = body_of_text[:24]

# 显示文章并保存
print('\n'.join(body_of_text))
with open('desktop/ST_gunLaws.txt', 'w') as f:
    f.write('\n'.join(body_of_text))

#————————————————————————————————————————————————————————————————————————下载NLTK需要用到的包————————————————————————————————————————————————————————————————————————
# 使用自然语言处理功能包之前需要下载模块需要用到的数据
import nltk
nltk.download('punkt')

#—————————————————————————————————————————————————————————————————————————标记化和标准化——————————————————————————————————————————————————————————————————
'''有了文章只是第一步，分析内容前，我们需要将文章拆成句子，进而拆成词
标准化就是讲不同时态、语态背景的词转换成普通形式'''
import nltk
from nltk.book import *

guns_laws = 'desktop/ST_gunLaws.txt'

with open(guns_laws, 'r') as f:
    article = f.read()

# 加载NLTK模块
sentencer = nltk.data.load('tokenizers/punkt/english.pickle')  # sentencer对象是一个punkt句子标记器。标记器使用无监督学习找到句子的始末，但没法处理句子中的缩写
tokenizer = nltk.RegexpTokenizer(r'\w+')  # RegexpTokenizer()以正则表达式作为参数，\w+只保留单词，但对省略识别不好
stemmer = nltk.PorterStemmer()  # PorterStemmer()根据特定算法移除单词的结尾部分(如reading变成read)，从而还原为词干
lemmatizer = nltk.WordNetLemmatizer()  # 还原词也是为了标准化文本，不同于词干的是，还原词使用巨大的词库，找出单词的语言学意义上的词干

# 将文本拆成句子
sentences = sentencer.tokenize(article)

words = []  # 词集
stemmed_words = []  # 词干集
lemmatized_words = []  # 还原词集

# 遍历所有句子
for sentence in sentences:
    # 将句子拆成词
    words.append(tokenizer.tokenize(sentence))
    # 提取词干
    stemmed_words.append([stemmer.stem(word) for word in words[-1]])
    # 还原词形
    lemmatized_words.append([lemmatizer.lemmatize(word) for word in words[-1]])

# 存储各词集
file_words  = 'desktop/ST_gunLaws_words.txt'
file_stems  = 'desktop/ST_gunLaws_stems.txt'
file_lemmas = 'desktop/ST_gunLaws_lemmas.txt'

# 对保存的词集逐个分行
with open(file_words, 'w') as f:
    for w in words:
        for word in w:
            f.write(word + '\n')

with open(file_stems, 'w') as f:
    for w in stemmed_words:
        for word in w:
            f.write(word + '\n')

with open(file_lemmas, 'w') as f:
    for w in lemmatized_words:
        for word in w:
            f.write(word + '\n')

#—————————————————————————————————————————————————————————————————————————识别词类与命名实体——————————————————————————————————————————————————————————————————
import nltk
import re

## 方法一
def preprocess_data(text):   # 进行句和词的标记化
    global sentences, tokenized
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    sentences =  nltk.sent_tokenize(text)
    tokenized = [tokenizer.tokenize(s) for s in sentences]

guns_laws = 'desktop/ST_gunLaws.txt'

with open(guns_laws, 'r') as f:
    article = f.read()

# 分割成句并标记
sentences = []
tokenized = []
words = []

preprocess_data(article)

# 标记词类
tagged_sentences = [nltk.pos_tag(s) for s in tokenized]  # pos_tag()查看整个句子
print(tagged_sentences)

# 提取出命名实体 -- 常规方法
named_entities = []

for sentence in tagged_sentences:
    for word in sentence:
        if word[1] == 'NNP' or word[1] == 'NNPS':
            named_entities.append(word)

named_entities = list(set(named_entities))

print('Named entities -- simplistic approach:')
print(named_entities)

# 提取出命名实体 -- 正则表达式做法
named_entities = []
tagged = []

pattern = 'ENT: {<DT>?(<NNP|NNPS>)+}'
tokenizer = nltk.RegexpParser(pattern)  #  使用正则表达式解析器

for sent in tagged_sentences:
    tagged.append(tokenizer.parse(sent))

for sentence in tagged:
    for pos in sentence:
        if type(pos) == nltk.tree.Tree:
            named_entities.append(pos)

named_entities = list(set([tuple(e) for e in named_entities]))

print('\nNamed entities using regular expressions:')
print(named_entities)

##方法二  使用NLTK提供的.ne_chunk_sents()
'''
此部分完全同上

def preprocess_data(text):    # 进行句和词的标记化
    global sentences, tokenized
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    sentences =  nltk.sent_tokenize(text)
    tokenized = [tokenizer.tokenize(s) for s in sentences]

guns_laws = 'desktop/ST_gunLaws.txt'

with open(guns_laws, 'r') as f:
    article = f.read()

# 分割成句并标记
sentences = []
tokenized = []
words = []

preprocess_data(article)

# 标记词类
tagged_sentences = [nltk.pos_tag(s) for s in tokenized]

'''
# 提取命名实体————命名实体切块
ne = nltk.ne_chunk_sents(tagged_sentences)

# 得到去重的列表
named_entities = []

for s in ne:
    for ne in s:
        if type(ne) == nltk.tree.Tree:
            named_entities.append((ne.label(), tuple(ne)))
        
named_entities = list(set(named_entities))
named_entities = sorted(named_entities)

# 打印列表
for t, ne in named_entities:
    print(t, ne)

#—————————————————————————————————————————————————————————————————————————识别文章主题——————————————————————————————————————————————————————————————————
import nltk
import re
import numpy as np
import matplotlib.pyplot as plt
'''
此部分完全同上
def preprocess_data(text):
    global sentences, tokenized
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    sentences =  nltk.sent_tokenize(text)
    tokenized = [tokenizer.tokenize(s) for s in sentences]

# import the data
guns_laws = '../../Data/Chapter09/ST_gunLaws.txt'

with open(guns_laws, 'r') as f:
    article = f.read()

# chunk into sentences and tokenize
sentences = []
tokenized = []

preprocess_data(article)

'''
# 标记词类
tagged_sentences = [nltk.pos_tag(w) for w in tokenized]

# 提取命名实体 -- 正则表达式法
tagged = []

pattern = '''
    ENT: {<DT>?(<NNP|NNPS>)+}
'''

tokenizer = nltk.RegexpParser(pattern)

for sent in tagged_sentences:
    tagged.append(tokenizer.parse(sent))

# 将命名实体列表用空格w[0]连在一起
words = []
lemmatizer = nltk.WordNetLemmatizer()

for sentence in tagged:
    for pos in sentence:
        if type(pos) == nltk.tree.Tree:
            words.append(' '.join([w[0] for w in pos]))
        else:
            words.append(lemmatizer.lemmatize(pos[0]))

# 移除停用词(the,a,and,in……)
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words if w.lower() not in stopwords]

# 计算词频
freq = nltk.FreqDist(words)  # .FreqDist()统计单词

# 根据词频降序排列
f = sorted(freq.items(), key=lambda x: x[1], reverse=True)

# 打印高频词
top_words = [w for w in f if w[1] > 1]
print(top_words, len(top_words))

# 绘图展示
top_words_transposed = list(zip(*top_words))
y_pos = np.arange(len(top_words_transposed[0][:10]))[::-1]

plt.barh(y_pos, top_words_transposed[1][:10], align='center', alpha=0.5)
plt.yticks(y_pos, top_words_transposed[0][:10])
plt.xlabel('Frequency')
plt.ylabel('Top words')
plt.savefig('desktop/word_frequency.png', dpi=300)


