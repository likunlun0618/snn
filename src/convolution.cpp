#include <iostream>
#include <sstream>
#include <cblas.h>
#include <cstdlib>
#include "convolution.h"
//#include "global.h"
//#include "time.h"

Module* createConvolution(std::string options)
{
    // 目前没有做错误处理
    // b=0表示添加bias，b=1表示不添加bias

    // 将所有的逗号替换为空格
    // 因为原始文件需要用空格做split，所有不能直接用带空格的字符串做options
    for (char &ch : options)
        ch = ch == ',' ? ' ' : ch;

    int c, n, k, s, p, b;
    std::stringstream(options) >> c >> n >> k >> s >> p >> b;
    Convolution *conv;
    if (b != 0)
        conv = new Convolution(c, n, k, s, p, true);
    else
        conv = new Convolution(c, n, k, s, p, false);
    return (Module*)conv;
}

Convolution::Convolution(int _c, int _n, int _k, int _s, int _p, bool _bias): s(_s), p(_p)
{
    weight.resize(_n, _c, _k, _k);
    if (_bias)
        bias.resize(1, 1, 1, _n);
}

Convolution::~Convolution() {}

Tensor Convolution::forward(const std::vector<Tensor> &inp)
{
    long t1, t2;

    // t1 = time();

    if (inp[0].c != weight.c)
    {
        ; // 输入和weight的channel不符
    }

    // p表示padding，s表示步长
    // weight.h == weight.w == kernel size
    int outh = (inp[0].h + 2*p - weight.h + s) / s;
    int outw = (inp[0].w + 2*p - weight.w + s) / s;

    // 向storage申请了足量的内存
    Tensor out(1, weight.n, outh, outw);

    // t1 = system_time();
    // 当bias为空Tensor的时候，bias.data为NULL
    _convolution(
        inp[0].data, weight.data, out.data, bias.data, \
        weight.c, inp[0].h, inp[0].w, weight.n, weight.h, s, p \
    );

    // t2 = time();
    /*
    if (global == 1 && global_module == 9)
    {
        std::cout << weight.h << "," << s << "," << p << std::endl;
        std::cout << t2 - t1 << std::endl;
    }
    //*/
    return out;
}

int Convolution::parameters() const
{
    if (bias.n * bias.c * bias.h * bias.w > 0)
        return 2;
    else
        return 1;
}

// index == 0表示加载weight的参数
// index == 1表示加载bias的参数
int Convolution::load(float *data, int size, int index)
{
    if (index == 0 && size != weight.n * weight.c * weight.h * weight.w)
        return -1; // weight长度不符合

    if (index == 1 && bias.n * bias.c * bias.h * bias.w <= 0)
        return -2; // 此卷积层没有偏置，但却尝试给偏置赋值

    if (index == 1 && size != bias.n * bias.c * bias.h * bias.w)
        return -3; // bias长度不符合

    if (index > 1)
        return -4; // 卷积的参数最多只有两个

    // TODO：改为memcpy
    float *p = index == 0 ? weight.data : bias.data;
    for (int i = 0; i < size; ++i)
        p[i] = data[i];

    return 0;
}

int Convolution::_convolution(float *inp, float *weight, float *out, float *bias, \
                              int c, int h, int w, int n, int k, int s, int p)
{
    //long t1 ,t2;
    //t1 = time();

    int outh = (h + 2*p - k + s) / s;
    int outw = (w + 2*p - k + s) / s;

    float *tmp;
    // trick
    if (k == 1 && s == 1 && p == 0)
    {
        tmp = inp;
    }
    else
    {
        tmp = new float[c * k * k * outh * outw];
        _im2col(inp, tmp, c, h, w, k, s, p);
    }

    //t2 = time();

    // 偏置相关
    float beta;
    if (bias)
    {
        for (int i = n - 1; i >= 0; --i)
        {
            float *p_out = out + i * (outh * outw);
            for (int j = outh * outw - 1; j >= 0; --j)
                p_out[j] = bias[i];
        }
        beta = 1.;
    }
    else
        beta = 0.;

    

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, outh*outw, c*k*k, 1., \
                weight, c*k*k, tmp, outh*outw, beta, out, outh*outw);

    if (k == 1 && s == 1 && p == 0)
        return 0;

    delete [] tmp;

    /*
    if (global == 1 && global_module == 31)
    {
        std::cout << k << "," << s << "," << p << std::endl;
        std::cout << t2 - t1 << "us" << std::endl;
    }
    */

    return 0;
}

int Convolution::_im2col(float *inp, float *out, int c, int h, int w, int k, int s, int p)
{
    int outh = (h + 2*p - k + s) / s;
    int outw = (w + 2*p - k + s) / s;

    #pragma omp parallel for num_threads(4)
    for (int ci = 0; ci < c; ++ci)
    {
        float *input = inp + ci * h * w;
        int offset = ci * k * k * outh * outw;
        for (int kh = 0; kh < k; ++kh)
            for (int kw = 0; kw < k; ++kw)
                for (int i = 0; i < outh; ++i)
                    for (int j = 0; j < outw; ++j)
                    {
                        int row = kh + i * s - p;
                        int col = kw + j * s - p;
                        if (row >= 0 && row < h && col >= 0 && col < w)
                            out[offset++] = input[row * w + col];
                        else
                            out[offset++] = 0.;
                    }
    }

    return 0;
}

void Convolution::mergeBN(float *k, float *b, int size)
{
    if (size != weight.n)
    {
        std::cout << "BatchNorm size != Convolution weight n" << std::endl;
        exit(0);
    }

    // 假如当前的卷积层没有bias
    if (bias.n * bias.c * bias.h * bias.w <= 0)
    {
        bias.resize(1, 1, 1, size);
        for (int i = 0; i < size; ++i)
            bias.data[i] = 0.;
    }

    // k2c表示k^2 * c，其中k是kernel size，c是channel
    int k2c = weight.c * weight.h * weight.w;
    for (int i = 0; i < size; ++i)
    {
        // 融合BN层的weight
        for (int j = 0; j < k2c; ++j)
        {
            weight.data[i * k2c + j] *= k[i];
        }
        // 融合BN层的bias
        bias.data[i] = bias.data[i] * k[i] + b[i];
    }
}
