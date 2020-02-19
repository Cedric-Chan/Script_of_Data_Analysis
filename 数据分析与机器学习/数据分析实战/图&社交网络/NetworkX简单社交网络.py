import networkx as nx
import networkx.algorithms as alg  # 提供了各种图算法
import numpy as np
import matplotlib.pyplot as plt

# 创建图对象
twitter = nx.Graph()

# 加入用户节点
twitter.add_node('Tom', age= 34)
twitter.add_node('Rachel', age= 33)
twitter.add_node('Skye', age= 29)
twitter.add_node('Bob', age= 45)
twitter.add_node('Mike', age=23)
twitter.add_node('Peter', age=46)
twitter.add_node('Matt', age=58)
twitter.add_node('Lester', age=65)
twitter.add_node('Jack', age= 32)
twitter.add_node('Max', age= 75)
twitter.add_node('Linda', age= 23)
twitter.add_node('Rory', age= 18)
twitter.add_node('Richard', age= 24)
twitter.add_node('Jackie', age= 25)
twitter.add_node('Alex', age= 24)
twitter.add_node('Bart', age= 33)
twitter.add_node('Greg', age= 45)
twitter.add_node('Rob', age= 19)
twitter.add_node('Markus',age= 21)
twitter.add_node('Glenn', age= 24)

# 加入发帖数
twitter.node['Rory']['posts'] = 182
twitter.node['Rob']['posts'] = 111
twitter.node['Markus']['posts'] = 159
twitter.node['Linda']['posts'] = 128
twitter.node['Mike']['posts'] = 289
twitter.node['Alex']['posts'] = 188
twitter.node['Glenn']['posts'] = 252
twitter.node['Richard']['posts'] = 106
twitter.node['Jackie']['posts'] = 138
twitter.node['Skye']['posts'] = 78
twitter.node['Jack']['posts'] = 62
twitter.node['Bart']['posts'] = 38
twitter.node['Rachel']['posts'] = 89
twitter.node['Tom']['posts'] = 23
twitter.node['Bob']['posts'] = 21
twitter.node['Greg']['posts'] = 41
twitter.node['Peter']['posts'] = 64
twitter.node['Matt']['posts'] = 8
twitter.node['Lester']['posts'] = 4
twitter.node['Max']['posts'] = 2

# 添加共同粉丝（谁认识谁）
twitter.add_edge('Rob', 'Rory', Weight=1)
twitter.add_edge('Markus', 'Rory', Weight= 1)
twitter.add_edge('Markus', 'Rob', Weight= 5)
twitter.add_edge('Mike', 'Rory', Weight= 1)
twitter.add_edge('Mike', 'Rob', Weight= 1)
twitter.add_edge('Mike', 'Markus',Weight= 1)
twitter.add_edge('Mike', 'Linda', Weight= 5)
twitter.add_edge('Alex', 'Rob', Weight= 1)
twitter.add_edge('Alex', 'Markus',Weight= 1)
twitter.add_edge('Alex', 'Mike', Weight= 1)
twitter.add_edge('Glenn', 'Rory', Weight= 1)
twitter.add_edge('Glenn', 'Rob', Weight= 1)
twitter.add_edge('Glenn', 'Markus', Weight= 1)
twitter.add_edge('Glenn', 'Linda', Weight= 2)
twitter.add_edge('Glenn', 'Mike', Weight= 1)
twitter.add_edge('Glenn', 'Alex', Weight= 1)
twitter.add_edge('Richard', 'Rob', Weight= 1)
twitter.add_edge('Richard', 'Linda', Weight= 1)
twitter.add_edge('Richard', 'Mike', Weight= 1)
twitter.add_edge('Richard', 'Alex', Weight= 1)
twitter.add_edge('Richard', 'Glenn', Weight= 1)
twitter.add_edge('Jackie', 'Linda', Weight= 1)
twitter.add_edge('Jackie', 'Mike', Weight= 1)
twitter.add_edge('Jackie', 'Glenn', Weight= 1)
twitter.add_edge('Jackie', 'Skye', Weight= 1)
twitter.add_edge('Tom', 'Rachel', Weight= 5)
twitter.add_edge('Rachel', 'Bart', Weight= 1)
twitter.add_edge('Tom', 'Bart', Weight= 2)
twitter.add_edge('Jack', 'Skye', Weight= 1)
twitter.add_edge('Bart', 'Skye', Weight= 1)
twitter.add_edge('Rachel', 'Skye', Weight= 1)
twitter.add_edge('Greg', 'Bob', Weight= 1)
twitter.add_edge('Peter', 'Greg', Weight= 1)
twitter.add_edge('Lester', 'Matt', Weight= 1)
twitter.add_edge('Max', 'Matt', Weight= 1)
twitter.add_edge('Rachel', 'Linda', Weight= 1)
twitter.add_edge('Tom', 'Linda', Weight= 1)
twitter.add_edge('Bart', 'Greg', Weight=2)
twitter.add_edge('Tom', 'Greg', Weight= 2)
twitter.add_edge('Peter', 'Lester', Weight= 2)
twitter.add_edge('Tom', 'Mike', Weight= 1)
twitter.add_edge('Rachel', 'Mike', Weight= 1)
twitter.add_edge('Rachel', 'Glenn', Weight= 1)
twitter.add_edge('Lester', 'Max', Weight= 1)
twitter.add_edge('Matt', 'Peter', Weight= 1)

# 添加关系
twitter['Rob']['Rory']['relationship'] = 'friend'
twitter['Markus']['Rory']['relationship'] = 'friend'
twitter['Markus']['Rob']['relationship'] = 'spouse'
twitter['Mike']['Rory']['relationship'] = 'friend'
twitter['Mike']['Rob']['relationship'] = 'friend'
twitter['Mike']['Markus']['relationship'] = 'friend'
twitter['Mike']['Linda']['relationship'] = 'spouse'
twitter['Alex']['Rob']['relationship'] = 'friend'
twitter['Alex']['Markus']['relationship'] = 'friend'
twitter['Alex']['Mike']['relationship'] = 'friend'
twitter['Glenn']['Rory']['relationship'] = 'friend'
twitter['Glenn']['Rob']['relationship'] = 'friend'
twitter['Glenn']['Markus']['relationship'] = 'friend'
twitter['Glenn']['Linda']['relationship'] = 'sibling'
twitter['Glenn']['Mike']['relationship'] = 'friend'
twitter['Glenn']['Alex']['relationship'] = 'friend'
twitter['Richard']['Rob']['relationship'] = 'friend'
twitter['Richard']['Linda']['relationship'] = 'friend'
twitter['Richard']['Mike']['relationship'] = 'friend'
twitter['Richard']['Alex']['relationship'] = 'friend'
twitter['Richard']['Glenn']['relationship'] = 'friend'
twitter['Jackie']['Linda']['relationship'] = 'friend'
twitter['Jackie']['Mike']['relationship'] = 'friend'
twitter['Jackie']['Glenn']['relationship'] = 'friend'
twitter['Jackie']['Skye']['relationship'] = 'friend'
twitter['Tom']['Rachel']['relationship'] = 'spouse'
twitter['Rachel']['Bart']['relationship'] = 'friend'
twitter['Tom']['Bart']['relationship'] = 'sibling'
twitter['Jack']['Skye']['relationship'] = 'friend'
twitter['Bart']['Skye']['relationship'] = 'friend'
twitter['Rachel']['Skye']['relationship'] = 'friend'
twitter['Greg']['Bob']['relationship'] = 'friend'
twitter['Peter']['Greg']['relationship'] = 'friend'
twitter['Lester']['Matt']['relationship'] = 'friend'
twitter['Max']['Matt']['relationship'] = 'friend'
twitter['Rachel']['Linda']['relationship'] = 'friend'
twitter['Tom']['Linda']['relationship'] = 'friend'
twitter['Bart']['Greg']['relationship'] = 'sibling'
twitter['Tom']['Greg']['relationship'] = 'sibling'
twitter['Peter']['Lester']['relationship'] = 'generation'
twitter['Tom']['Mike']['relationship'] = 'friend'
twitter['Rachel']['Mike']['relationship'] = 'friend'
twitter['Rachel']['Glenn']['relationship'] = 'friend'
twitter['Lester']['Max']['relationship'] = 'friend'
twitter['Matt']['Peter']['relationship'] = 'friend'

# 显示节点信息
print('\nJust nodes: ', twitter.nodes())
print('\nNodes with data: ', twitter.nodes(data=True))

# 显示边信息
print('\nEdges with data: ', twitter.edges(data=True))

# 图密度与度数
'''
.density()度量图中节点之间的连通度：所有节点都与剩余全部节点相连时密度为1.
图的密度就是图中边的数目与可能的边的数目的比值（可能的边的数目=n*(n-1)/2）
'''
print('\nDensity of the graph: ', nx.density(twitter))  

'''
节点的度数是其所有邻居的数目，即邻接节点的总数
.centrality.degree_centrality()计算的是节点的度数与图中最大可能度数（节点数-1）的比值
'''
centrality = sorted(alg.centrality.degree_centrality(twitter).items(),key=lambda e: e[1], reverse=True)  # 根据中心度降序排列，越靠前联系越多。即Mike和Glenn是图中连接最多的人
print('\nCentrality of nodes: ', centrality)

'''
.assortativity.average_neighbor_degree()计算各节点的邻居的平均度数，可显示出谁在网络中连接了更多的人
'''
average_degree = sorted(alg.assortativity.average_neighbor_degree(twitter).items(), key=lambda e: e[1], reverse=True)  # Rory,Jackie,Richard居于中心，是扩展网络的最优选择
print('\nAverage degree: ', average_degree)
#print(len(twitter['Glenn']) / 19)

# 初步作图（内容有重叠，即使完全相同的连接，也很有可能具有不同的布局）
nx.draw_networkx(twitter)
plt.savefig('desktop/twitter_networkx.png')

# 保存为Gephi格式
nx.write_graphml(twitter, 'desktop/twitter.graphml')