#ifndef ADAPTIVE_AVGPOOL_H
#define ADAPTIVE_AVGPOOL_H

#include "module.h"

namespace snn
{

Module* createAdaptiveAvgPool(std::string options);

class AdaptiveAvgPool : public Module
{
public:
    AdaptiveAvgPool(int h, int w);

    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;

private:
    int outh, outw;
};

} // namespace snn

#endif  // ADAPTIVE_AVGPOOL_H
