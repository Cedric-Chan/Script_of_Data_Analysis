import nltk

def print_tree(tree, filename):
    '''
        将解析出的NLTK树保存至PS文件
    '''
    # 创建画布
    canvasFrame = nltk.draw.util.CanvasFrame()
    # 创建挂件
    widget = nltk.draw.TreeWidget(canvasFrame.canvas(), tree)
    # 将挂件放在画布上
    canvasFrame.add_widget(widget, 10, 10)
    # 保存文件
    canvasFrame.print_to_file(filename)
    # 释放对象所占内存
    canvasFrame.destroy()

# 取两个句子
sentences = ['Washington state voters last fall passed Initiative 594', 
'The White House also said it planned to ask Congress for $500 million to improve mental health care, and Obama issued a memorandum directing federal agencies to conduct or sponsor research into smart gun technology that reduces the risk of accidental gun discharges.']

# 简单的标记器（在每个空格处截断）
sentences = [s.split() for s in sentences]

# 标记词类
sentences = [nltk.pos_tag(s) for s in sentences]

# 匹配句子结构的模式
pattern = '''
  NP: {<DT|JJ|NN.*|CD>+}   # NP包括一个名词和其修饰语。具体由DT(限定词the,a)、JJ(形容词)、CD(数量词)或名词的变种组成。NN.*表示NN(单数名词)、NNS(复数名词)、NNP(可数名词)
  VP: {<VB.*><NP|PP>+}     # VP由各种形式的动词组成:VB(基本型)、VBD(过去时)、VBG(动名词或现在进行时)、VBN(过去进行时)、VBP(非第三人称的单数现在形式)、VBZ(第三人称单数形式)，还有NP或PP
  PP: {<IN><NP>}           # PP是介词和NP的组合
'''

# 识别出块
NPChunker = nltk.RegexpParser(pattern)
chunks = [NPChunker.parse(s) for s in sentences]

# save to file
print_tree(chunks[0], 'desktop/sent1.ps')
print_tree(chunks[1], 'desktop/sent2.ps')