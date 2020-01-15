#include <sstream>
#include "maxpool.h"

namespace snn
{

Module* createMaxPool(std::string options)
{
    // 将所有的逗号替换为空格
    // 因为原始文件需要用空格做split，所有不能直接用带空格的字符串做options
    for (char &ch : options)
        ch = ch == ',' ? ' ' : ch;

    int k = 0, s = 0, p = 0;
    std::stringstream(options) >> k >> s >> p;

    if (s == 0)
        s = k;

    MaxPool *maxpool = new MaxPool(k, s, p);
    return (Module*)maxpool;
}

MaxPool::MaxPool(int _k, int _s, int _p): k(_k), s(_s), p(_p) {}

Tensor MaxPool::forward(const std::vector<Tensor> &inp)
{
    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;
    int outh = (h + 2*p - k + s) / s;
    int outw = (w + 2*p - k + s) / s;
    Tensor out(1, c, outh, outw);

    if (k == s && p == 0)
    {
        // trick
        #pragma omp parallel for num_threads(4)
        for (int ci = 0; ci < c; ++ci)
        {
            float *input = inp[0].data + ci * h * w;
            float *output = out.data + ci * outh * outw;
            int n = 0;
            for (int hi = 0; hi < outh; ++hi)
                for (int wi = 0; wi < outw; ++wi)
                {
                    float max = input[hi*s*w + wi*s];
                    for (int i = 0; i < k; ++i)
                        for (int j = 0; j < k; ++j)
                        {
                            float tmp = input[(hi * s + i) * w + (wi * s + j)];
                            if (tmp > max)
                                max = tmp;
                        }
                    output[n++] = max;
                }
        }
    }
    else
    {
        // general case
        #pragma omp parallel for num_threads(4)
        for (int ci = 0; ci < c; ++ci)
        {
            float *input = inp[0].data + ci * h * w;
            float *output = out.data + ci * outh * outw;
            int n = 0;
            for (int hi = 0; hi < outh; ++hi)
                for (int wi = 0; wi < outw; ++wi)
                {
                    float max;
                    int row = hi * s - p;
                    int col = wi * s - p;
                    if (row >= 0 && row < h && col >= 0 && col < w)
                        max = input[row * w + col];
                    else
                        max = 0.;

                    for (int i = 0; i < k; ++i)
                        for (int j = 0; j < k; ++j)
                        {
                            float tmp;
                            row = i + hi * s - p;
                            col = j + wi * s - p;
                            if (row >= 0 && row < h && col >= 0 && col < w)
                                tmp = input[row * w + col];
                            else
                                tmp = 0.;
                            if (tmp > max)
                                max = tmp;
                        }
                    output[n++] = max;
                }
        }
    }

    return out;
}

int MaxPool::parameters() const
{
    return 0;
}

//int MaxPool::load(float *data, int size, int index) {}

} // namespace snn
