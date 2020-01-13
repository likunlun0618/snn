#include <sstream>
#include <cblas.h>
#include "linear.h"

namespace snn
{

Module* createLinear(std::string options)
{
    for (char &ch : options)
        ch = ch == ',' ? ' ' : ch;
    int c, n, b;
    std::stringstream(options) >> c >> n >> b;
    Linear *linear;
    if (b != 0)
        linear = new Linear(c, n, true);
    else
        linear = new Linear(c, n, false);
    return (Module*)linear;
}

Linear::Linear(int _c, int _n, bool _bias): c(_c), n(_n)
{
    weight.resize(1, 1, n, c);
    if (_bias)
        bias.resize(1, 1, 1, n);
}

Linear::~Linear() {}

Tensor Linear::forward(const std::vector<Tensor> &inp)
{
    if (inp[0].w != c)
    {
        ; // 维度不符
    }

    Tensor out(1, 1, 1, n);

    float beta;
    if (bias.data)
    {
        for (int i = 0; i < n; ++i)
            out.data[i] = bias.data[i];
        beta = 1.;
    }
    else
        beta = 0.;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, 1, c, 1., \
                weight.data, c, inp[0].data, 1, beta, out.data, 1);

    return out;
}

int Linear::parameters() const
{
    if (bias.data)
        return 2;
    else 
        return 1;
}

int Linear::load(float *data, int size, int index)
{
    if (index == 0 && size != n * c)
        return -1;
    if (index == 1 && !bias.data)
        return -2;
    if (index == 1 && size != n)
        return -3;
    if (index > 1)
        return -4;

    float *p = index == 0 ? weight.data : bias.data;
    for (int i = 0; i < size; ++i)
        p[i] = data[i];

    return 0;
}

} // namespace snn
