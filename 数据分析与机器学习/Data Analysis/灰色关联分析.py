import pandas as pd
import numpy as np

def gra_one(data_frame, m=0):
    gray = data_frame
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    # 标准要素，iloc取列
    std = gray.iloc[:, m]
    # 比较要素
    ce = gray.iloc[:, 0:]
    n = ce.shape[0]
    m = ce.shape[1]

    # 与标准要素比较，相减
    a = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            a[i, j] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中最大值和最小值
    c = np.amax(a)
    d = np.amin(a)

    # 计算值
    result = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            result[i, j] = (d+0.5*c)/(a[i, j]+0.5*c)
    # 求均值，得到灰色关联值
    result2 = np.zeros(m)
    for i in range(m):
        result2[i] = np.mean(result[i, :])
    RT = pd.DataFrame(result2)
    return RT

def gra(data_frame):
    list_columns = [str(s) for s in range(len(data_frame.columns)) if s not in [None]]
    df_local = pd.DataFrame(columns=list_columns)
    for i in range(len(data_frame.columns)):
        # 按列进行的操作
        df_local.iloc[:, i] = gra_one(data_frame, i)[0]
    return df_local

'''if __name__ == '__main__'的意思是当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
    当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行'''
if __name__ == '__main__':
    dxy = pd.read_csv(r'C:\Users\123\Desktop\123.csv')
    dxy_gra = gra(dxy)
    dxy_gra.to_csv("C:/Users/123/Desktop/Grey_Related.csv")