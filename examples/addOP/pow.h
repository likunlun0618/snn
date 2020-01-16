#ifndef POW_H
#define POW_H

#include <module.h>

snn::Module* createPow(std::string options);

class Pow : public snn::Module
{
public:
    Pow(float _index);
    snn::Tensor forward(const std::vector<snn::Tensor> &inp);
    int parameters() const;

private:
    float index;
};

#endif
