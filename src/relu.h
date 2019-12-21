#ifndef RELU_H
#define RELU_H

#include "module.h"

Module* createReLU(std::string options);

class ReLU : public Module
{
public:
    // ReLU暂时不需要初始化
    // ReLU();
    // ~ReLU();

    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;

    // 对ReLU来说这是个空函数
    // int load(float *data, int size, int index);
};

#endif  // RELU_H
