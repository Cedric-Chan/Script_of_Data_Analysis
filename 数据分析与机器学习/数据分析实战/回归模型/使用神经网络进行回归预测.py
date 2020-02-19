# 导入helper脚本
import sys
sys.path.append('C:/Users/123/Desktop/work')
import helper as hlp

import pandas as pd 
import pybrain.structure as st
import pybrain.supervised.trainers as tr
import pybrain.tools.shortcuts as pb

@hlp.timeit
def fitANN(data):
    '''
        构建神经网络回归器
    '''
    inputs_cnt = data['input'].shape[1]
    target_cnt = data['target'].shape[1]
    # 创建回归器对象
    ann = pb.buildNetwork(inputs_cnt,  # 一个输入神经元
        inputs_cnt * 3,   # 三个隐藏层神经元
        target_cnt,  # 一个输出神经元
        hiddenclass=st.TanhLayer,  # 隐藏层为tanh激活函数
        outclass=st.LinearLayer,  # 输出层为线性激活函数
        bias=True  # 含偏差（常数）项
    )
    # 创建训练对象
    trainer = tr.BackpropTrainer(ann, data, verbose=True, batchlearning=False)  # 显示日志，batchlearning=False使用在线训练
    # 训练网络
    trainer.trainUntilConvergence(maxEpochs=50, verbose=True, continueEpochs=2, validationProportion=0.25)  # validationProportion=0.25使用四分之一的数据用于验证
    # 返回回归器
    return ann

r_filename = 'desktop/power_plant_dataset_pc.csv'
csv_read = pd.read_csv(r_filename)

train_x, train_y, test_x,  test_y, labels = hlp.split_data(csv_read, y='net_generation_MWh', x=['total_fuel_cons_mmbtu'])

# 创建神经网络训练集和测试集
training = hlp.prepareANNDataset((train_x, train_y), prob='regression')
testing  = hlp.prepareANNDataset((test_x, test_y), prob='regression')

# 训练模型
regressor = fitANN(training)
# 预测数据
predicted = regressor.activateOnDataset(testing)
# 计算R方
score = hlp.get_score(test_y, predicted[:, 0])
print('R2: ', score)