import struct
import numpy as np
import torch


def write_array(p, filename):
    '''
    p是numpy数组的list，其中每个numpy数组的维度在1~4之间
    '''

    # Tensor的个数
    p = p.copy()
    raw = struct.pack('i', len(p))

    for i in range(len(p)):
        # 维度的长度
        raw += struct.pack('i', 4)

        # 维度
        if len(p[i].shape) == 1:
            shape = (1, 1, 1, p[i].shape[0])
        elif len(p[i].shape) == 2:
            shape = (1, 1, p[i].shape[1], p[i].shape[0])
        elif len(p[i].shape) == 3:
            shape = (1, p[i].shape[2], p[i].shape[1], p[i].shape[0])
        else:
            shape = p[i].shape
        s = 1
        for j in range(4):
            s *= shape[j]
            raw += struct.pack('i', shape[j])

        p[i] = p[i].reshape(-1)
        p[i] = p[i].tolist()

        # 数据
        raw += struct.pack('f'*len(p[i]), *(p[i]))

    with open(filename, 'wb') as f:
        f.write(raw)


p = []
p.append(np.load('tmp_out.npy'))


write_array(p, 'tmp_out.array')
