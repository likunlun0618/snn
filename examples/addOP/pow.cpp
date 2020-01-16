#include <iostream>
#include <sstream>
#include <cmath>
#include "pow.h"

snn::Module* createPow(std::string options)
{
    float index = 0.;
    std::stringstream(options) >> index;
    Pow *pow = new Pow(index);
    return (snn::Module*)pow;
}

Pow::Pow(float _index): index(_index) {}

snn::Tensor Pow::forward(const std::vector<snn::Tensor> &inp)
{
    int n = inp[0].n;
    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;
    snn::Tensor out(n, c, h, w);

    int size = n * c * h * w;
    for (int i = 0; i < size; ++i)
        out.data[i] = pow(inp[0].data[i], index);

    return out;
}

int Pow::parameters() const
{
    return 0;
}
