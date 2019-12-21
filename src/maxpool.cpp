#include <sstream>
#include "maxpool.h"

Module* createMaxPool(std::string options)
{
    int stride;
    std::stringstream(options) >> stride;
    MaxPool *maxpool = new MaxPool(stride);
    return (Module*)maxpool;
}

MaxPool::MaxPool(int _stride): stride(_stride) {}

Tensor MaxPool::forward(const std::vector<Tensor> &inp)
{
    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;
    int outh = h / stride;
    int outw = w / stride;
    Tensor out(1, c, outh, outw);

    #pragma omp parallel for num_threads(4)
    for (int ci = 0; ci < c; ++ci)
    {
        float *input = inp[0].data + ci * h * w;
        float *output = out.data + ci * outh * outw;
        int n = 0;
        for (int hi = 0; hi < outh; ++hi)
            for (int wi = 0; wi < outw; ++wi)
            {
                float max = input[hi*stride*w + wi*stride];
                for (int i = 0; i < stride; ++i)
                    for (int j = 0; j < stride; ++j)
                    {
                        float tmp = input[(hi * stride + i) * w + (wi * stride + j)];
                        if (tmp > max)
                            max = tmp;
                    }
                output[n++] = max;
            }
    }

    return out;
}

int MaxPool::parameters() const
{
    return 0;
}

//int MaxPool::load(float *data, int size, int index) {}
