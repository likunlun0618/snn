#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "module.h"

Module* createConvolution(std::string options);

class Convolution : public Module
{
public:
    // 根据c，n，k，bias创建weight和bias
    // 不对weight和bias进行初始化
    Convolution(int _c, int _n, int _k, int _s, int _p, bool _bias);

    // 要拷贝_weight和_bias的内容到weight和bias上，而不是直接给指针赋值
    // 假设_weight和_bias是合法的
    //Convolution(float *_weight, int _c, int _n, int _k, int _s, int _p);
    //Convolution(float *_weight, float *_bias, int _c, int _n, int _k, int _s, int _p);

    // 使用Tensor的话，就不自己释放内存
    ~Convolution();

    Tensor forward(const std::vector<Tensor> &inp);

    int parameters() const;

    int load(float *data, int size, int index);
    //int load(std::string filename, int index);

    // 将BN层的参数合并到weight和bias中
    // 当卷积层的下一个节点只有BN的时候，就可以执行这个操作
    // 合并之后的卷积效率没有任何变化，可以实现BN层的0开销
    //void mergeBN(Tensor running_mean, Tensor running_var, Tensor alpha, Tensor beta, float eps);
    void mergeBN(float *k, float *b, int size);

private:
    Tensor weight;
    Tensor bias;

    int s;  // stride
    int p;  // padding

    // 将输入展开成矩阵
    int _im2col(float *inp, float *out, int c, int h, int w, int k,int s, int p);

    // 利用openBlas的矩阵乘法实现卷积运算
    int _convolution(float *inp, float *weight, float*out, float*bias, \
                      int c, int h, int w, int n, int k, int s, int p);  
};

#endif  // CONVOLUTION_H
