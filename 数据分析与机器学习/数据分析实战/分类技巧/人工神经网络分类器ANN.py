'''
神经元网络的训练，其实是神经元之间联结权重的变更，与每个神将元激活函数参数的调整
最流行的监督学习范式是误差反向传播训练方法
'''
# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

#————————————————————————————————————————————————————————————————————————单一隐藏层——————————————————————————————————————————————————————————————————
#  
import pandas as pd 
import pybrain.structure as st   # .structure让我们可以访问多种激活函数
import pybrain.supervised.trainers as tr   # 为网络提供了监督方法
import pybrain.tools.shortcuts as pb   # 允许我们快速构建网络

@hlp.timeit
def fitANN(data):
    '''
        构建人工神经网络分类器
    '''
    # 确定输入输出的数目（输入数量是数据集的列数，输出是因变量的层数）
    inputs_cnt = data['input'].shape[1]  # .shape获取列的数量
    target_cnt = data['target'].shape[1]

    # 创建分类器对象
    ann = pb.buildNetwork(inputs_cnt,  # 一参是输入层神经元数目
        inputs_cnt * 2,   # 二参是隐藏层神经元数目（可接受任意层数的隐藏层）
        target_cnt,       # 最后一个参是输出层神经元数目
        hiddenclass=st.TanhLayer,  # 为隐藏层指定TanhLayer激活函数（tanh将输入压缩到0-1内，形状类似S函数，但优于S函数）
        outclass=st.SoftmaxLayer,  # 为输出层指定SoftmaxLayer激活函数
        bias=True  # 在求和函数中加入一个常数偏差项
    )

    # 通过tr.BackpropTrainer反向传播算法创建训练器对象
    trainer = tr.BackpropTrainer(ann, data,  # 一参是新创建的网络，二参是数据集
        verbose=True, batchlearning=False)   # 记录训练进度，关闭批量学习 （批量学习是可并行的填鸭式，在线学习是基于样本的逐个调整参数和误差）

    # 训练网络（trainUntilConvergence表示运行到收敛为止）
    trainer.trainUntilConvergence(maxEpochs=50, verbose=True,  # 最大迭代50次，记录进度
        continueEpochs=3, validationProportion=0.25)  # 收敛后继续迭代3次，取训练集的1/4来验证模型

    # 返回分类器
    return ann

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter03/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 拆分数据
train_x, train_y, \
test_x,  test_y, \
labels = hlp.split_data(
    csv_read, 
    y = 'credit_application',
    x = ['n_duration','n_euribor3m','n_age','n_emp_var_rate','n_pdays','month_mar','prev_ctc_outcome_success','n_cons_price_idx','month_apr','n_cons_conf_idx']
)

# 创建训练集与测试集
training = hlp.prepareANNDataset((train_x, train_y))
testing  = hlp.prepareANNDataset((test_x,  test_y))

# 训练模型
classifier = fitANN(training)

# 分类新数据
predicted = classifier.activateOnDataset(testing)  # activateOnDataset()方法输入测试数据集

# 最低的激活函数输出给类别
predicted = predicted.argmin(axis=1)  # 通过argmin()找出最小值的输出的下标，作为归类

# 结果输出
hlp.printModelSummary(test_y, predicted)

#————————————————————————————————————————————————————————————————————————典型双隐藏层——————————————————————————————————————————————————————————————————
import helper as hlp
import pandas as pd 
import pybrain.datasets as dt
import pybrain.structure as st
import pybrain.supervised.trainers as tr
import pybrain.tools.shortcuts as pb
import pybrain.utilities as ut

@hlp.timeit
def fitANN(data):
    '''
        构建人工神经网络分类器
    '''
    # 确定输入输出的数目（输入数量是数据集的列数，输出是因变量的层数）
    inputs_cnt = data['input'].shape[1]  # .shape获取列的数量
    target_cnt = data['target'].shape[1]

    ann = pb.buildNetwork(inputs_cnt, # 输入神经元数目（10个）
        inputs_cnt * 2,   # 第一隐藏层神经元数目（20个）
        int(inputs_cnt / 2),   # 第二隐藏层神经元数目（5个）,除法默认的float型不被识别，需转为int型
        target_cnt,  # 输出层神经元数目
        hiddenclass=st.SigmoidLayer,  # 隐藏层激活函数SigmoidLayer
        outclass=st.SoftmaxLayer,  # 输出层激活函数SoftmaxLayer
        bias=True  # 加入常数偏差项
    )

    # 通过tr.BackpropTrainer反向传播算法创建训练器对象
    trainer = tr.BackpropTrainer(ann, data,  # 一参是新创建的网络，二参是数据集
        verbose=True, batchlearning=False)   # 记录训练进度，关闭批量学习 （批量学习是可并行的填鸭式，在线学习是基于样本的逐个调整参数和误差）

    # 训练网络（trainUntilConvergence表示运行到收敛为止）
    trainer.trainUntilConvergence(maxEpochs=50, verbose=True,  # 最大迭代50次，记录进度
        continueEpochs=3, validationProportion=0.25)  # 收敛后继续迭代3次，取训练集的1/4来验证模型

    # 返回分类器
    return ann

r_filename = 'C:/Users/123/Desktop/work/PracticalCookbook/Data/Chapter03/bank_contacts.csv'
csv_read = pd.read_csv(r_filename)

# 拆分数据
train_x, train_y, \
test_x,  test_y, \
labels = hlp.split_data(
    csv_read, 
    y = 'credit_application',
    x = ['n_duration','n_euribor3m','n_age','n_emp_var_rate','n_pdays','month_mar','prev_ctc_outcome_success','n_cons_price_idx','month_apr','n_cons_conf_idx']
)

# 创建训练集与测试集
training = hlp.prepareANNDataset((train_x, train_y))
testing  = hlp.prepareANNDataset((test_x,  test_y))

# 训练模型
classifier = fitANN(training)

# 分类新数据
predicted = classifier.activateOnDataset(testing)  # activateOnDataset()方法输入测试数据集

# 最低的激活函数输出给类别
predicted = predicted.argmin(axis=1)  # 通过argmin()找出最小值的输出的下标，作为归类

# 结果输出
hlp.printModelSummary(test_y, predicted)




