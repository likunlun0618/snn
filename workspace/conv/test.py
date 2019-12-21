import time
import os
import struct
import numpy as np
import torch

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

config = read_array('config.array')
inp = read_array('inp.array')
fil = read_array('filter.array')
bias = read_array('bias.array')
out1 = read_array('out.array')

c = int(config[0])
h = int(config[1])
w = int(config[2])
n = int(config[3])
k = int(config[4])
s = int(config[5])
p = int(config[6])

model = torch.nn.Conv2d(c, n, k, s, p)
# model = torch.nn.Conv2d(c, n, k, s, p, bias=False).double()
model.weight.data = torch.from_numpy(fil)
model.bias.data = torch.from_numpy(bias)
img = torch.from_numpy(inp).unsqueeze(0)
# img = torch.from_numpy(inp).unsqueeze(0).double()

with torch.no_grad():
    model(img)
    model(img)
    t1 = time.time()
    out2 = model(img)
    t2 = time.time()
out2 = out2[0].numpy()

print('pytorch time:', (t2 - t1) * 1000)
print('input dims:', c, h, w)
print('kernel size:', k)
print('stride:', s)
print('padding:', p)
print('cpp out shape:', out1.shape)
print('pytorch out shape:', out2.shape)
print('average error:', abs(out1 - out2).mean())

os.system('rm *.array')
