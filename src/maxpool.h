#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "module.h"

namespace snn
{

Module* createMaxPool(std::string options);

class MaxPool : public Module
{
public:
    MaxPool(int _k, int _s, int _p);
    // 暂时不需要析构函数
    // ~MaxPool();

    // 此处始终假设inp和out是合法的指针
    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;

    //int load(float *data, int size, int index);

private:
    int k, s, p;
};

} // namespace snn

#endif  // MAXPOOL_H
