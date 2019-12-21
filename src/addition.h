#ifndef ADDITION_H
#define ADDITION_H

#include "module.h"

Module* createAddition(std::string options);

class Addition : public Module
{
public:
    Tensor forward(const std::vector<Tensor> &inp);
    int parameters() const;
};

#endif // ADDITION_H
