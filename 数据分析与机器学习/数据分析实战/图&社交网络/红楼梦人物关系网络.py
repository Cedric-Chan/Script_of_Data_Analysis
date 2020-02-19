# -*- coding: utf-8 -*-
import pandas as pd
from pyecharts.charts import Graph
from pyecharts import options as opts
import jieba
import jieba.posseg as pseg


def deal_data():

    # 读取数据并加载词典
    with open("desktop/hlm.txt", encoding='utf-8') as f:
        honglou = f.readlines()
    jieba.load_userdict("desktop/renwu_forcut.txt")   # 通过 load_userdict将自定义的词典加载到 jieba 库,dict.txt字典务必要保存成utf-8形式
    renwu_data = pd.read_csv("desktop/renwu_forcut.txt",  header=None)
    mylist = [k[0].split(" ")[0] for k in renwu_data.values.tolist()]
    
    # 对文本进行分词处理并提取 
    tmpNames = []   # 存在于我们自定义词典的人名，保存到一个临时变量当中 tmpNames
    names = {}
    relationships = {}
    for h in honglou:
        h.replace("贾妃", "元春")  # 文中"贾妃", "元春"，"李宫裁", "李纨" 混用严重，所以这里直接做替换处理
        h.replace("李宫裁", "李纨")
        poss = pseg.cut(h)  # 使用 jieba 库提供的 pseg 工具来做分词处理，会返回每个分词的词性
        tmpNames.append([])
        for w in poss:
            if w.flag != 'nr' or len(w.word) != 2 or w.word not in mylist:  # 只有符合要求且在我们提供的字典列表里的分词，才会保留
                continue
            tmpNames[-1].append(w.word)
            if names.get(w.word) is None:
                names[w.word] = 0
            relationships[w.word] = {}
            names[w.word] += 1  # 一个人每出现一次，就会增加一
    print(relationships)
    print(tmpNames)
    for name, times in names.items():
        print(name, times)
    
    # 处理人物关系
    for name in tmpNames:
        for name1 in name:
            for name2 in name:
                if name1 == name2:
                    continue
                if relationships[name1].get(name2) is None:
                    relationships[name1][name2] = 1
                else:
                    relationships[name1][name2] += 1  # 出现在同一个段落中的人物，我们认为他们是关系紧密的，每同时出现一次，关系增加1
    print(relationships)

    # 保存到文件
    with open("desktop/relationship.csv", "w", encoding='utf-8') as f:  # 文件1：人物关系表，包含首先出现的人物、之后出现的人物和一同出现次数
        f.write("Source,Target,Weight\n")
        for name, edges in relationships.items():
            for v, w in edges.items():
                f.write(name + "," + v + "," + str(w) + "\n")

    with open("desktop/NameNode.csv", "w", encoding='utf-8') as f:  # 文件2：人物比重表，包含该人物总体出现次数，出现次数越多，认为所占比重越大
        f.write("ID,Label,Weight\n")
        for name, times in names.items():
            f.write(name + "," + name + "," + str(times) + "\n")


# 制作关系图表
def deal_graph():
    relationship_data = pd.read_csv('desktop/relationship.csv')
    namenode_data = pd.read_csv('desktop/NameNode.csv')
    relationship_data_list = relationship_data.values.tolist()
    namenode_data_list = namenode_data.values.tolist()

    nodes = []
    for node in namenode_data_list:
        if node[0] == "宝玉":
            node[2] = node[2]/3
        nodes.append({"name": node[0], "symbolSize": node[2]/30})
    links = []
    for link in relationship_data_list:
        links.append({"source": link[0], "target": link[1], "value": link[2]})

    g = (
        Graph()
        .add("", nodes, links, repulsion=8000)
        .set_global_opts(title_opts=opts.TitleOpts(title="红楼人物关系"))
    )
    return g


if __name__ == '__main__':
    deal_data()
    g = deal_graph()
    g.render()