#include <iostream>
#include <sys/timeb.h>
#include <cblas.h>
#include "random.h"
#include "io.h"

using namespace std;

long system_time()
{
    timeb t;
    ftime(&t);
    return t.time * 1000 + t.millitm;
}

/*
c, h, w分别是输入的三个维度
k是kernel size
s是stride
p是padding
out是(outh * outw) * (k^2 * c)的二维矩阵
*/
void im2col(float *inp, float *out, int c, int h, int w, int k, int s, int p)
{
    int outh = (h + 2*p - k + s) / s;
    int outw = (w + 2*p - k + s) / s;
        #pragma omp parallel for num_threads(4)
        for (int ci = 0; ci < c; ++ci)
        {
            float *input = inp + ci * h * w;
            int bias = ci * k * k * outh * outw;
            for (int kh = 0; kh < k; ++kh)
                for (int kw = 0; kw < k; ++kw)
                    for (int i = 0; i < outh; ++i)
                        for (int j = 0; j < outw; ++j)
                        {
                            int row = kh + i * s - p;
                            int col = kw + j * s - p;
                            if (row >= 0 && row < h && col >= 0 && col < w)
                                // out[n++] = inp[ci * h * w + row * w + col];
                                out[bias++] = input[row * w + col];
                            else
                                out[bias++] = 0.;
                        }
        }
}

/*
c, h, w分别是输入的3个维度
n是filter的个数
k是kernel size
s是stride
p是padding
*/
void conv(float *inp, float *filter, float *out, float *bias, int c, int h, int w, int n, int k, int s, int p)
{
    int outh = (h + 2*p - k + s) / s;
    int outw = (w + 2*p - k + s) / s;

    float *tmp;
    // 这里使用了trick，如果k==1,s==1,p==0，说明im2col的结果在内存上和inp完全一致，所以省去了赋值一遍的开销
    if (k == 1 && s == 1 && p == 0)
        tmp = inp;
    else {
        tmp = new float[c * k * k * outh * outw];
        im2col(inp, tmp, c, h, w, k, s, p);
    }

    float beta;
    if (bias) {
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

    //*
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, outh*outw, c*k*k, 1., \
                filter, c*k*k, tmp, outh*outw, beta, out, outh*outw);
    //*/

    // 目前的实验结果是，relu采用下面这种循环的实现，在ms量级为0
    //*
    int out_size = n * outh * outw;
    for (int i = 0; i < out_size; ++i)
        if (out[i] < 0.)
            out[i] = 0.;
    //*/

    // trick，因为k==1,s==1,p==0的时候，没有给tmp开辟额外的空间，所以不需要释放
    if (k == 1 && s == 1 && p == 0)
        return;

    delete[] tmp;
}

int main()
{
    seed();

    // int c = 128, h = 64, w = 64, n = 128, k = 3, s = 1, p = k / 2;
    int c = 3, h = 256, w = 256, n = 64, k = 7, s = 2, p = k / 2;

    float *inp = new float[c * h * w];
    for (int i = c * h * w - 1; i >= 0; --i)
        inp[i] = randf();

    float *filter = new float[n * c * k * k];
    for (int i = n * c * k * k - 1; i >= 0; --i)
        filter[i] = randf();

    float *bias = new float[n];
    for (int i = 0; i < n; ++i)
        bias[i] = randf();

    int outh = (h + 2*p - k + s) / s;
    int outw = (w + 2*p - k + s) / s;
    float *out = new float[n * outh * outw];

    long t1 = system_time();
    conv(inp, filter, out, bias, c, h, w, n, k, s, p);
    long t2 = system_time();
    cout << "cpp time: " << t2 - t1 << endl;

    float config[7] = {(float)c, (float)h, (float)w, (float)n, (float)k, (float)s, (float)p};
    write_array("config.array", config, {7});
    write_array("inp.array", inp, {c, h, w});
    write_array("filter.array", filter, {n, c, k, k});
    write_array("bias.array", bias, {n});
    write_array("out.array", out, {n, outh, outw});

    return 0;
}
