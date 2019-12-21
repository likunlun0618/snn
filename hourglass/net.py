import torch
import torch.nn as nn
import numpy as np
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


class Residual(nn.Module):

    def __init__(self, inp, out):
        super(Residual, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.ReLU(),
            nn.Conv2d(inp, out//2, 1, 1, 0),
            nn.BatchNorm2d(out//2),
            nn.ReLU(),
            nn.Conv2d(out//2, out//2, 3, 1, 1),
            nn.BatchNorm2d(out//2),
            nn.ReLU(),
            nn.Conv2d(out//2, out, 1, 1, 0)
        )
        if inp == out:
            self.skip_layer = None
        else:
            self.skip_layer = nn.Conv2d(inp, out, 1, 1, 0)

    def forward(self, x):
        if self.skip_layer is None:
            return self.conv_block(x) + x
        else:
            return self.conv_block(x) + self.skip_layer(x)


class Hourglass(nn.Module):

    def __init__(self, f, n):
        super(Hourglass, self).__init__()
        self.upper_branch = Residual(f, f)
        self.lower_branch = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            Residual(f, f),
            Residual(f, f) if n == 1 else Hourglass(f, n-1),
            Residual(f, f),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.upper_branch(x) + self.lower_branch(x)


class StackedHourglass(nn.Module):

    def __init__(self, out_channels, f, n, stacks):
        super(StackedHourglass, self).__init__()
        self.stacks = stacks
        self.pre_module = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Residual(64, f//2),
            nn.MaxPool2d(2, stride=2),
            Residual(f//2, f//2),
            Residual(f//2, f)
        )
        self.hg = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        self.middle_branch = nn.ModuleList()
        self.output_branch = nn.ModuleList()
        for i in range(stacks):
            self.hg.append(nn.Sequential(
                Hourglass(f, n),
                Residual(f, f),
                nn.Conv2d(f, f, 1, 1, 0),
                nn.BatchNorm2d(f),
                nn.ReLU()
            ))
            self.out_layer.append(nn.Conv2d(f, out_channels, 1, 1, 0))
            if i < stacks-1:
                self.middle_branch.append(nn.Conv2d(f, f, 1, 1, 0))
                self.output_branch.append(nn.Conv2d(out_channels, f, 1, 1, 0))

    def forward(self, x):
        y = []
        x = self.pre_module(x)
        for i in range(self.stacks):
            middle = self.hg[i](x)
            output = self.out_layer[i](middle)
            y.append(output)
            if i < self.stacks-1:
                x = x + self.middle_branch[i](middle) + self.output_branch[i](output)
        return y


def append_residual(p, model, skip):
    p.append(model.conv_block[0].running_mean.data)
    p.append(model.conv_block[0].running_var.data)
    p.append(model.conv_block[0].weight.data)
    p.append(model.conv_block[0].bias.data)

    p.append(model.conv_block[2].weight.data)
    p.append(model.conv_block[2].bias.data)

    p.append(model.conv_block[3].running_mean.data)
    p.append(model.conv_block[3].running_var.data)
    p.append(model.conv_block[3].weight.data)
    p.append(model.conv_block[3].bias.data)

    p.append(model.conv_block[5].weight.data)
    p.append(model.conv_block[5].bias.data)

    p.append(model.conv_block[6].running_mean.data)
    p.append(model.conv_block[6].running_var.data)
    p.append(model.conv_block[6].weight.data)
    p.append(model.conv_block[6].bias.data)

    p.append(model.conv_block[8].weight.data)
    p.append(model.conv_block[8].bias.data)

    if skip:
        p.append(model.skip_layer.weight.data)
        p.append(model.skip_layer.bias.data)

def append_hg(p, model):
    append_residual(p, model[0].lower_branch[1], False)
    append_residual(p, model[0].lower_branch[2].lower_branch[1], False)
    append_residual(p, model[0].lower_branch[2].lower_branch[2].lower_branch[1], False)
    append_residual(p, model[0].lower_branch[2].lower_branch[2].lower_branch[2].lower_branch[1], False)
    append_residual(p, model[0].lower_branch[2].lower_branch[2].lower_branch[2].lower_branch[2], False)
    append_residual(p, model[0].lower_branch[2].lower_branch[2].lower_branch[2].lower_branch[3], False)
    append_residual(p, model[0].lower_branch[2].lower_branch[2].lower_branch[2].upper_branch, False)
    append_residual(p, model[0].lower_branch[2].lower_branch[2].lower_branch[3], False)
    append_residual(p, model[0].lower_branch[2].lower_branch[2].upper_branch, False)
    append_residual(p, model[0].lower_branch[2].lower_branch[3], False)
    append_residual(p, model[0].lower_branch[2].upper_branch, False)
    append_residual(p, model[0].lower_branch[3], False)
    append_residual(p, model[0].upper_branch, False)
    append_residual(p, model[1], False)

    p.append(model[2].weight.data)
    p.append(model[2].bias.data)

    p.append(model[3].running_mean.data)
    p.append(model[3].running_var.data)
    p.append(model[3].weight.data)
    p.append(model[3].bias.data)


if __name__ == '__main__':

    import time

    x = torch.rand(1, 3, 256, 256)
    write_array([x], 'input.array')

    print('start loop 0 ...')
    # input()

    model = StackedHourglass(16, 256, 4, 4)
    model.load_state_dict(torch.load('hg4.pth'))
    model.eval()

    with torch.no_grad():
        model(x)[-1]
        model(x)[-1]

        t1 = time.time()
        x = model(x)[-1]
        t2 = time.time()

    print('time:', (t2 - t1) * 1000)

    print(x.size())
    write_array([x], 'output.array')

    print('start loop 1 ...')
    # input()
