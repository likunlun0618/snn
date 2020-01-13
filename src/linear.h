#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"

namespace snn
{

Module* createLinear(std::string options);

class Linear : public Module
{
public:
    Linear(int _c, int _n, bool _bias);

    ~Linear();

    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;
    int load(float *data, int size, int index);

private:
    Tensor weight;
    Tensor bias;

    int c;
    int n;
};

} // namespace snn

#endif
