#ifndef BN_H
#define BN_H

#include "module.h"

Module* createBatchNorm(std::string options);

class BatchNorm : public Module
{
public:
    BatchNorm(int _c, float _eps);

    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;

    int load(float *data, int size, int index);

private:
    float *mean;
    float *var;
    Tensor weight;
    Tensor bias;
    // 记录已经有多少个参数被加载
    // 当4个参数已经全部被加载的时候，将4个参数融合成两个
    int count;

    int c;
    float eps;
};

#endif // BN_H
