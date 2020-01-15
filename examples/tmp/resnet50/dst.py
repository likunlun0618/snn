import os
import numpy as np
import torch
import torch.nn as nn
import struct


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


def bn_helper(bn, path):
    bn.weight.data.copy_(torch.load(path + 'weight.pth'))
    bn.bias.data.copy_(torch.load(path + 'bias.pth'))
    bn.running_mean.copy_(torch.load(path + 'running_mean.pth'))
    bn.running_var.copy_(torch.load(path + 'running_var.pth'))


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU()

        if inplanes != planes * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )
        else:
            self.downsample = None

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def load(self, dir):
        self.conv1.weight.data.copy_(torch.load(os.path.join(dir, 'conv1-weight.pth')))
        bn_helper(self.bn1, os.path.join(dir, 'bn1-'))
        self.conv2.weight.data.copy_(torch.load(os.path.join(dir, 'conv2-weight.pth')))
        bn_helper(self.bn2, os.path.join(dir, 'bn2-'))
        self.conv3.weight.data.copy_(torch.load(os.path.join(dir, 'conv3-weight.pth')))
        bn_helper(self.bn3, os.path.join(dir, 'bn3-'))

        if self.downsample is not None:
            self.downsample[0].weight.data.copy_(torch.load(os.path.join(dir, 'downsample-conv-weight.pth')))
            bn_helper(self.downsample[1], os.path.join(dir, 'downsample-bn-'))


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # layer1
        self.layer1_block1 = Bottleneck(64, 64, downsample=True)
        self.layer1_block2 = Bottleneck(256, 64)
        self.layer1_block3 = Bottleneck(256, 64)

        # layer2
        self.layer2_block1 = Bottleneck(256, 128, stride=2, downsample=True)
        self.layer2_block2 = Bottleneck(512, 128)
        self.layer2_block3 = Bottleneck(512, 128)
        self.layer2_block4 = Bottleneck(512, 128)

        # layer3
        self.layer3_block1 = Bottleneck(512, 256, stride=2, downsample=True)
        self.layer3_block2 = Bottleneck(1024, 256)
        self.layer3_block3 = Bottleneck(1024, 256)
        self.layer3_block4 = Bottleneck(1024, 256)
        self.layer3_block5 = Bottleneck(1024, 256)
        self.layer3_block6 = Bottleneck(1024, 256)

        self.layer4_block1 = Bottleneck(1024, 512, stride=2, downsample=True)
        self.layer4_block2 = Bottleneck(2048, 512)
        self.layer4_block3 = Bottleneck(2048, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)

        self.load()

    def load(self):
        self.conv1.weight.data.copy_(torch.load('para/conv1-weight.pth'))
        bn_helper(self.bn1, 'para/bn1-')

        self.layer1_block1.load('para/layer1-block1/')
        self.layer1_block2.load('para/layer1-block2/')
        self.layer1_block3.load('para/layer1-block3/')

        self.layer2_block1.load('para/layer2-block1/')
        self.layer2_block2.load('para/layer2-block2/')
        self.layer2_block3.load('para/layer2-block3/')
        self.layer2_block4.load('para/layer2-block4/')

        self.layer3_block1.load('para/layer3-block1/')
        self.layer3_block2.load('para/layer3-block2/')
        self.layer3_block3.load('para/layer3-block3/')
        self.layer3_block4.load('para/layer3-block4/')
        self.layer3_block5.load('para/layer3-block5/')
        self.layer3_block6.load('para/layer3-block6/')

        self.layer4_block1.load('para/layer4-block1/')
        self.layer4_block2.load('para/layer4-block2/')
        self.layer4_block3.load('para/layer4-block3/')

        self.fc.weight.data.copy_(torch.load('para/fc-weight.pth'))
        self.fc.bias.data.copy_(torch.load('para/fc-bias.pth'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        x = self.layer1_block3(x)

        x = self.layer2_block1(x)
        x = self.layer2_block2(x)
        x = self.layer2_block3(x)
        x = self.layer2_block4(x)

        x = self.layer3_block1(x)
        x = self.layer3_block2(x)
        x = self.layer3_block3(x)
        x = self.layer3_block4(x)
        x = self.layer3_block5(x)
        x = self.layer3_block6(x)

        x = self.layer4_block1(x)
        x = self.layer4_block2(x)
        x = self.layer4_block3(x)


        x = self.avgpool(x)
        
        x = torch.flatten(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':

    inp = np.load('inp.npy')
    inp = torch.from_numpy(inp)
    out1 = np.load('out.npy')
    out1 = torch.from_numpy(out1)

    model = Model()
    model.eval()

    with torch.no_grad():
        out2 = model(inp)

    print((out1 - out2).abs().mean())

    print(out2.max())
    print(out2.argmax())
