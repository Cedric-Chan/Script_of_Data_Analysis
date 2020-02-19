# 改关系矩阵和对比矩阵数值后直接F5运行即可得出候选对象的最终返回值

import numpy as np

RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

def get_w(array):
    row = array.shape[0]  # 计算出阶数
    a_axis_0_sum = array.sum(axis=0)
    # print(a_axis_0_sum)
    b = array / a_axis_0_sum  # 新的矩阵b
    # print(b)
    b_axis_0_sum = b.sum(axis=0)
    b_axis_1_sum = b.sum(axis=1)  # 每一行的特征向量
    # print(b_axis_1_sum)
    w = b_axis_1_sum / row  # 归一化处理(特征向量)
    nw = w * row
    AW = (w * array).sum(axis=1)
    # print(AW)
    max_max = sum(AW / (row * w))
    # print(max_max)
    CI = (max_max - row) / (row - 1)
    CR = CI / RI_dict[row]
    if CR < 0.1:
        # print(round(CR, 3))
        # print('满足一致性')
        # print(np.max(w))
        # print(sorted(w,reverse=True))
        # print(max_max)
        # print('特征向量:%s' % w)
        return w
    else:
        print(round(CR, 3))
        print('不满足一致性，请进行修改')
   

def main(array):
    if type(array) is np.ndarray:
        return get_w(array)
    else:
        print('请输入numpy对象')

# e为指标元素的两两比较判别矩阵
e = np.array(
    [
        [
            1, 3, 1/3, 1/7, 1/6
            ], 
        [
            1/3, 1, 1/5, 1/9, 1/8
            ],
        [
            3, 5, 1, 1/5, 1/3
            ],
        [
            7, 9, 5, 1, 4
            ],
        [
            6, 8, 3, 1/4, 1
            ]
    ]
)

main(e)                    # 返回即为指标元素的特征向量（即为对于上层母元素的权重，即为层次单排序）

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。

if __name__ == '__main__':
    # 曦笑月妍敏
    
    # character指标的对比矩阵
    a = np.array(
        [
            [
                1, 5, 9, 7, 3
                ], 
            [
                1/5, 1, 5, 3, 1/3
                ],
            [
                1/9, 1/5, 1, 1/3, 1/7
                ],
            [
                1/7, 1/3, 3, 1, 1/5
                ],
            [
                1/3, 3, 7, 5, 1
                ]
        ]
    )
    # height指标的对比矩阵
    b = np.array(
        [
            [
                1, 3, 5, 3, 1
                ], 
            [
                1/3, 1, 3, 1, 1/3
                ],
            [
                1/5, 1/3, 1, 1/3, 1/5
                ],
            [
                1/3, 1, 3, 1, 1/3
                ],
            [
                1, 3, 5, 3, 1
                ]
        ]
    )
    # outlook指标的对比矩阵
    c = np.array(
        [
            [
                1, 6, 3, 8, 4
                ], 
            [
                1/6, 1, 1/4, 3, 1/3
                ],
            [
                1/3, 4, 1, 5, 2
                ],
            [
                1/8, 1/3, 1/5, 1, 1/4
                ],
            [
                1/4, 3, 1/2, 4, 1
                ]
        ]
    )
    # asset指标的对比矩阵
    d = np.array(
        [
            [
                1, 7, 9, 3, 5
                ], 
            [
                1/7, 1, 4, 1/6, 1/3
                ],
            [
                1/9, 1/4, 1, 1/8, 1/5
                ],
            [
                1/3, 6, 8, 1, 4
                ],
            [
                1/5, 3, 5, 1/4, 1
                ]
        ]
    )
    # edu指标的对比矩阵
    f = np.array(
        [
            [
                1, 1/3, 1/4, 1/5, 1/7
                ], 
            [
                3, 1, 1/3, 1/4, 1/5
                ],
            [
                4, 3, 1, 1/3, 1/4
                ],
            [
                5, 4, 3, 1, 1/3
                ],
            [
                7, 5, 4, 3, 1
                ]
        ]
    )
    e = main(e)
    a = main(a)
    b = main(b)
    c = main(c)
    d = main(d)
    f = main(f)
    try:
        res = np.array([a, b, c, d, f])
        ret = (np.transpose(res) * e).sum(axis=1)
        print(ret)
    except TypeError:
        print('数据有误，可能不满足一致性，请进行修改')