#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include "bn.h"

Module* createBatchNorm(std::string options)
{
    for (char &ch : options)
        ch = ch == ',' ? ' ' : ch;

    int c;
    float eps;
    std::stringstream(options) >> c >> eps;
    BatchNorm *bn = new BatchNorm(c, eps);
    return (Module*)bn;
}

BatchNorm::BatchNorm(int _c, float _eps): c(_c), eps(_eps), count(0), mean(NULL), var(NULL)
{
    // 错误处理
    // bn层的维度应该大于等于1，eps应该大于0
    if (c < 1)
    {
        std::cout << "BatchNorm error: c < 1" << std::endl;
        exit(-1);
    }
    if (eps < 0.)
    {
        std::cout << "BatchNorm error: eps < 0." << std::endl;
        exit(-1);
    }

    // running_mean.resize(1, 1, 1, c);
    // running_var.resize(1, 1, 1, c);
    weight.resize(1, 1, 1, c);
    bias.resize(1, 1, 1, c);
}

Tensor BatchNorm::forward(const std::vector<Tensor> &inp)
{
    // count != 4说明参数没有全部装载
    /*
    if (count != 4)
    {
        std::cout << "BatchNorm error: count != 4" << std::endl;
        exit(-1);
    }
    //*/

    // 因为已经提前将running_mean/running_var和weight/bias进行合并
    // 所以此处只进行weight * inp + bias的计算
    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;

    Tensor out(1, c, h, w);

    int s = h * w;
    #pragma omp parallel for num_threads(4)
    for (int ci = 0; ci < c; ++ci)
    {
        float *input = inp[0].data + ci * h * w;
        float *output = out.data + ci * h * w;
        float w = weight.data[ci];
        float b = bias.data[ci];
        for (int i = 0; i < s; ++i)
            output[i] = input[i] * w + b;
    }

    return out;
}

int BatchNorm::parameters() const
{
    return 4;
}

int BatchNorm::load(float *data, int size, int index)
{
    if (size != c)
    {
        std::cout << "BatchNorm error: size != c" << std::endl;
        exit(-1);
    }

    float *dst;
    if (index == 0)
    {
        mean = new float[size];
        dst = mean;
    }
    else if (index == 1)
    {
        var = new float[size];
        dst = var;
    }
    else if (index == 2)
        dst = weight.data;
    else if (index == 3)
        dst = bias.data;
    else
    {
        std::cout << "BatchNorm error: index > 3" << std::endl;
        exit(-1);
    }

    // TODO：改为memcpy
    for (int i = 0; i < c; ++i)
        dst[i] = data[i];

    ++count;
    // 提前将4个参数进行计算，变成两个参数
    // forward的时候计算过程就变成了weight*inp + bias
    if (count == 4 && mean != NULL && var != NULL)
    {
        for (int i = 0; i < c; ++i)
        {
            float k = weight.data[i] / sqrt(var[i] + eps);
            float b = bias.data[i] - mean[i] * k;
            weight.data[i] = k;
            bias.data[i] = b;
        }
        delete [] mean;
        delete [] var;
    }

    return 0;
}
