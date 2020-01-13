#ifndef ADDITION_H
#define ADDITION_H

#include "module.h"

namespace snn
{

Module* createAddition(std::string options);

class Addition : public Module
{
public:
    Tensor forward(const std::vector<Tensor> &inp);
    int parameters() const;
};

} // namespace snn

#endif // ADDITION_H
