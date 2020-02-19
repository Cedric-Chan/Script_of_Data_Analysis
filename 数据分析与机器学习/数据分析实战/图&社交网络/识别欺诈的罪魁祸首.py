import networkx as nx
import numpy as np
import collections as c

graph_file = 'desktop/fraud.gz'
fraud = nx.read_graphml(graph_file)

print('\nType of the graph: ', type(fraud))   # 显示图的类型（有并行边的有向图）

# 节点和边
nodes = fraud.nodes()  # 调出所有节点

nodes_population = [n for n in nodes if 'p_' in n]  # 买家节点的前缀是p_
nodes_merchants  = [n for n in nodes if 'm_' in n]  # 卖家节点的前缀是m_

n_population = len(nodes_population)
n_merchants  = len(nodes_merchants)
print('\nTotal population: {0}, number of merchants: {1}'.format(n_population, n_merchants))   # 显示节点列表长度

# 交易数目
n_transactions = fraud.number_of_edges()
print('Total number of transactions: {0}'.format(n_transactions))  # 显示交易总数（边数）

# what do we know about a transaction
p_1_transactions = fraud.out_edges('p_1', data=True)  # out_edges()获取p_1的全部交易
print('\nMetadata for a transaction: ', list(p_1_transactions))

print('Total value of all transactions: {0}'.format(np.sum([t[2]['amount'] for t in fraud.edges(data=True)])))  # 显示交易总金额

# 辨别信用卡泄露的消费者
all_disputed_transactions = [dt for dt in fraud.edges(data=True) if dt[2]['disputed']]

print('Total number of disputed transactions: {0}'.format(len(all_disputed_transactions)))  # 欺诈交易的数量
print('Total value of disputed transactions: {0}'.format(np.sum([dt[2]['amount'] for dt in all_disputed_transactions])))  # 欺诈涉及金额

# 受害者列表
people_scammed = list(set([p[0] for p in all_disputed_transactions]))  # set()生成一个去重的列表

print('Total number of people scammed: {0}'.format(len(people_scammed)))  # 受害者人数

# 所有异常交易列表
print('All disputed transactions:')

for dt in sorted(all_disputed_transactions, key=lambda e: e[0]):
    print('({0}, {1}: {{time:{2}, amount:{3}}})'.format(dt[0], dt[1], dt[2]['amount'], dt[2]['amount']))

# 每个人的损失
transactions = c.defaultdict(list)  # .defaultdict()类似字典

for p in all_disputed_transactions:
    transactions[p[0]].append(p[2]['amount'])

for p in sorted(transactions.items(), key=lambda e: np.sum(e[1]), reverse=True):  # 受害程度从大到小显示消费者列表
    print('Value lost by {0}: \t{1}'.format(p[0], np.sum(p[1])))

# 辨别出信用卡泄露的消费者
people_scammed = c.defaultdict(list)

for (person, merchant, data) in fraud.edges(data=True):
    if data['disputed']:
        people_scammed[person].append(data['time'])  

print('\nTotal number of people scammed: {0}'.format(len(people_scammed)))

# 每个受害者第一笔欺诈交易发生的时间
# scammed person
stolen_time = {} 

for person in people_scammed:
    stolen_time[person] = np.min(people_scammed[person])  # 找到受害者争议交易的最早时间

# 找出欺诈都涉及的卖家
merchants = c.defaultdict(list)
for person in people_scammed:
    edges = fraud.out_edges(person, data=True)
    for (person, merchant, data) in edges:
        if  stolen_time[person] - data['time'] <= 5 and stolen_time[person] - data['time'] >= 0:   # >=0用于找出第一次欺诈交易之前的所有交易，<=1用于回溯共同卖家的天数
            merchants[merchant].append(person)

merchants = [(merch, len(set(merchants[merch]))) for merch in merchants]  # 选出去重后的卖家

print('\nTop 5 merchants where people made purchases')
print('shortly before their credit cards were stolen')
print(sorted(merchants, key=lambda e: e[1], reverse=True)[:5])  # 所有33个受害者在第一笔欺诈的前一天都在4号卖家消费过



