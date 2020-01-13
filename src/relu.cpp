#include "relu.h"

namespace snn
{

Module* createReLU(std::string options)
{
    ReLU *relu = new ReLU;
    return (Module *)relu;
}

Tensor ReLU::forward(const std::vector<Tensor> &inp)
{
    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;
    Tensor out(1, c, h, w);

    int size = c * h * w;
    for (int i = 0; i < size; ++i)
    {
        float tmp = inp[0].data[i];
        out.data[i] = tmp < 0. ? 0. : tmp;
    }

    return out;
}

int ReLU::parameters() const
{
    return 0;
}

} // namespace snn

//int ReLU::load(float *data, int size, int index) {}
