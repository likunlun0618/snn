#ifndef UPSAMPLE_H
#define UPSAMPLE_H

#include "module.h"

namespace snn
{

Module* createUpsample(std::string options);

class Upsample : public Module
{
public:
    Upsample(int _scale);

    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;

private:
    int scale;
};

} // namespace snn

#endif  // UPSAMPLE_H
