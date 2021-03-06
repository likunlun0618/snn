#ifndef BN_H
#define BN_H

#include "module.h"

namespace snn
{

Module* createBatchNorm(std::string options);

class BatchNorm : public Module
{
public:
    BatchNorm(int _c, float _eps);

    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;

    int load(float *data, int size, int index);

    // 用于融合卷积和BN的接口
    float* ptrk() const;
    float* ptrb() const;
    int size() const;

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

} // namespace snn

#endif // BN_H
