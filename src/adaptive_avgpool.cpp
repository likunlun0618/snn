#include <sstream>
#include <cmath>
#include "adaptive_avgpool.h"

namespace snn
{

Module* createAdaptiveAvgPool(std::string options)
{
    // 将所有的逗号替换为空格
    // 因为原始文件需要用空格做split，所有不能直接用带空格的字符串做options
    for (char &ch : options)
        ch = ch == ',' ? ' ' : ch;

    int h = 0, w = 0;
    std::stringstream(options) >> h >> w;

    if (w == 0)
        w = h;

    AdaptiveAvgPool *adaptive = new AdaptiveAvgPool(h, w);
    return (Module*)adaptive;
}

AdaptiveAvgPool::AdaptiveAvgPool(int h, int w): outh(h), outw(w) {}

inline int start(int inp, int out, int i)
{
    return (int)std::floor((float)(inp * i) / out);
}

inline int end(int inp, int out, int i)
{
    return (int)std::ceil((float)(inp * (i+1)) / out);
}

Tensor AdaptiveAvgPool::forward(const std::vector<Tensor> &inp)
{
    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;
    Tensor out(1, c, outh, outw);

    if (outh == 1 && outw == 1)
    // if (false)
    {
        // special case
        // trick
        int s = h * w;
        for (int ci = 0; ci < c; ++ci)
        {
            float *input = inp[0].data + ci * s;

            float avg = 0.;
            for (int i = 0; i < s; ++i)
                avg += input[i];
            avg /= s;

            out.data[ci] = avg;
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
                    int starth = start(h, outh, hi);
                    int endh = end(h, outh, hi);
                    int startw = start(w, outw, wi);
                    int endw = end(w, outw, wi);

                    float avg = 0.;
                    for (int i = starth; i < endh; ++i)
                        for (int j = startw; j < endw; ++j)
                            avg += input[i * w + j];
                    avg /= ((endh - starth) * (endw - startw));

                    output[n++] = avg;
                }
        }
    }

    return out;
}

int AdaptiveAvgPool::parameters() const
{
    return 0;
}

} // namespace snn
