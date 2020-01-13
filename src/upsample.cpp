#include <sstream>
#include "upsample.h"

namespace snn
{

Module* createUpsample(std::string options)
{
    int scale;
    std::stringstream(options) >> scale;
    Upsample *upsample = new Upsample(scale);
    return (Module*)upsample;
}

Upsample::Upsample(int _scale): scale(_scale) {}

Tensor Upsample::forward(const std::vector<Tensor> &inp)
{
    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;
    int outh = h * scale;
    int outw = w * scale;
    Tensor out(1, c, outh, outw);

    #pragma omp parallel for num_threads(4)
    for (int ci = 0; ci < c; ++ci)
    {
        float *input = inp[0].data + ci * h * w;
        float *output = out.data + ci * outh * outw;
        int n = 0;
        for (int hi = 0; hi < outh; ++hi)
            for (int wi = 0; wi < outw; ++wi)
                output[n++] = input[hi / scale * w + wi / scale];
    }

    return out;
}

int Upsample::parameters() const
{
    return 0;
}

} // namespace snn
