import time
import os
import struct
import torch
import numpy as np


def read_array(filename):
    with open(filename, 'rb') as f:
        byte = f.read()

    # 读取维度
    dim = struct.unpack('i', byte[0:4])[0]
    dims = struct.unpack('i'*dim, byte[4:4*(dim+1)])

    # 读取数据
    length = 1
    for i in range(dim):
        length *= dims[i]

    data = struct.unpack('f'*length, byte[4*(dim+1):])
    # data = struct.unpack('d'*length, byte[4*(dim+1):])

    # 构造numpy数组
    array = np.array(data, dtype=np.float32)
    # array = np.array(data, dtype=np.float64)
    array = array.reshape(dims)
    return array


inp = read_array('inp.array')
out1 = read_array('out.array')

inp = torch.from_numpy(inp).unsqueeze(0)

model = torch.nn.MaxPool2d(2, stride=2)
with torch.no_grad():
    model(inp)[0].numpy()
    model(inp)[0].numpy()
    t1 = time.time()
    out2 = model(inp)[0].numpy()
    t2 = time.time()

print('pytorch time:', (t2 - t1) * 1000, 'ms')

print('average error:', abs(out1 - out2).mean())

os.system('rm *.array')
