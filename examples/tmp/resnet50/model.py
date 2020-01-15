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


def f(path):
    return torch.load(path).numpy()

def bn(p, path):
    p.append(f(path+'running_mean.pth'))
    p.append(f(path+'running_var.pth'))
    p.append(f(path+'weight.pth'))
    p.append(f(path+'bias.pth'))

def block(p, l, b, downsample=None):
    p.append(f('para/layer%d-block%d/conv1-weight.pth'%(l, b)))
    bn(p, 'para/layer%d-block%d/bn1-'%(l, b))
    p.append(f('para/layer%d-block%d/conv2-weight.pth'%(l, b)))
    bn(p, 'para/layer%d-block%d/bn2-'%(l, b))
    p.append(f('para/layer%d-block%d/conv3-weight.pth'%(l, b)))
    bn(p, 'para/layer%d-block%d/bn3-'%(l, b))
    if downsample is not None:
        p.append(f('para/layer%d-block%d/downsample-conv-weight.pth'%(l, b)))
        bn(p, 'para/layer%d-block%d/downsample-bn-'%(l, b))

p = []
p.append(f('para/conv1-weight.pth'))
bn(p, 'para/bn1-')

block(p, 1, 1, True)
block(p, 1, 2)
block(p, 1, 3)

block(p, 2, 1, True)
block(p, 2, 2)
block(p, 2, 3)
block(p, 2, 4)

block(p, 3, 1, True)
block(p, 3, 2)
block(p, 3, 3)
block(p, 3, 4)
block(p, 3, 5)
block(p, 3, 6)

block(p, 4, 1, True)
block(p, 4, 2)
block(p, 4, 3)

p.append(f('para/fc-weight.pth'))
p.append(f('para/fc-bias.pth'))

write_array(p, 'resnet50.array')
