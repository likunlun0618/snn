#ifndef RESHAPE_H
#define RESHAPE_H

#include "module.h"

namespace snn
{

Module* createReshape(std::string options);

class Reshape : public Module
{
public:
    Reshape(int _n, int _c, int _h, int _w);
    Tensor forward(const std::vector<Tensor> &inp);
    int parameters() const;
    //int load(float *data, int size, int index);

private:
    int n, c, h, w;
};

} // namespace snn

#endif
