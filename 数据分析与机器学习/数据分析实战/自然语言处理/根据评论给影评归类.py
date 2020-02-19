# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import nltk
import nltk.sentiment as sent
import json

@hlp.timeit
def classify_movies(train, sentim_analyzer):
    '''
        使用朴素贝叶斯分类器
        以根据评论归类电影
    '''
    nb_classifier = nltk.classify.NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(nb_classifier, train)
    return classifier

@hlp.timeit
def evaluate_classifier(test, sentim_analyzer):
    '''
        上面函数的补充
    '''
    for key, value in sorted(sentim_analyzer.evaluate(test).items()):
        print('{0}: {1}'.format(key, value))

f_training = 'desktop/movie_reviews_train.json'
f_testing  = 'desktop/movie_reviews_test.json'

# 读取json格式文件方法
with open(f_training, 'r') as f:
    read = f.read()
    train = json.loads(read)

with open(f_testing, 'r') as f:
    read = f.read()
    test = json.loads(read)

# 标记词
tokenizer = nltk.tokenize.TreebankWordTokenizer()  # 使用TreebankWordTokenizer()方法标记评论，不拆分评论，仅识别每条评论是正面还是负面

train = [(tokenizer.tokenize(r['review']), r['sentiment']) for r in train]   # 元组列表中，第一个元素r['review']是评论标记化的单词列表，第二个元素r['sentiment']是识别出的情感(pos或者neg)
test  = [(tokenizer.tokenize(r['review']), r['sentiment']) for r in test]

# 分析评论的情感
sentim_analyzer = sent.SentimentAnalyzer()  # SentimentAnalyzer()实现特征提取和分类
all_words_neg_flagged = sentim_analyzer.all_words([sent.util.mark_negation(doc) for doc in train])  # .util.mark_negation()处理单词列表，标出跟在否定形式之后的单词

# 获取最常见的词
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg_flagged, min_freq=4)  # 使用unigram_word_feats()得到频率高于min_freq的所有单词

# 添加特征提取器
sentim_analyzer.add_feat_extractor(sent.util.extract_unigram_feats, unigrams=unigram_feats)  # add_feat_extractor()为SentimentAnalyzer()对象从文档提取单词，且一参为extract_unigram_feats

# 用新创建的特征提取器转换训练集和测试集
train = sentim_analyzer.apply_features(train)  # apply_features()将单词列表转换成特征向量的列表，列表中的每个元素标明评论是否包含某个单词/特征
test  = sentim_analyzer.apply_features(test)

# 归类电影并计算分类器的性能
classify_movies(train, sentim_analyzer)  # 在开头的函数classify_movies()中可以改为贝叶斯之外的其他训练器
evaluate_classifier(test, sentim_analyzer)
