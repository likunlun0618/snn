#include <iostream>
#include <sys/timeb.h>
#include "random.h"
#include "io.h"

using namespace std;

long system_time()
{
    timeb t;
    ftime(&t);
    return t.time * 1000 + t.millitm;
}

// 最原始的卷积实现，使用了六重循环
// 不考虑步长、pad，步长始终为1，pad为0
// inp, filter, out分别为输入，卷积核，输出的内存的首地址，分别用一个线性的空间表示
// c1, h1, w1表示inp的维度
// n2表示卷积核的个数，也是输出的channel
// h2, w2表示卷积核的尺寸
// c2和c1必须相等，所以没有出现在参数中
// 输出的维度为n2 * h3 * w3
// 其中h3 = h1 - h2 + 1;
// 其中w3 = w1 - w2 + 1;
void naive_convolution(float *inp, float *filter, float *out,
                       int c1, int h1, int w1,
                       int n2, int h2, int w2)
{
    // inp的offset
    int offset1_c = h1 * w1, offset1_h = w1, offset1_w = 1;

    // filter的offset
    int c2 = c1;
    int offset2_n = c2 * h2 * w2, offset2_c = h2 * w2, offset2_h = w2, offset2_w = 1;

    // out的offset
    int h3 = h1 - h2 + 1;
    int w3 = w1 - w2 + 1;
    int offset3_c = h3 * w3, offset3_h = w3, offset3_w = 1;

    /*
    for (int i = n2 * h3 * w3 - 1; i >= 0; --i)
        out[i] = 0.;
    //*/

    for (int n = 0; n < n2; ++n)
        for (int h = 0; h + h2 <= h1; h += 1)
            for (int w = 0; w + w2 <= w1; w += 1) { 
                float s = 0.;
                for (int i = 0; i < c2; ++i)
                    for (int j = 0; j < h2; ++j)
                        for (int k = 0; k < w2; ++k)
                            s += inp[i * offset1_c + (h + j) * offset1_h + (w + k) * offset1_w] \
                                 * filter[n * offset2_n + i * offset2_c + j * offset2_h + k * offset2_w];
                out[n * offset3_c + h * offset3_h + w * offset3_w] = s;
            }
}

// 内存连续
void naive_convolution2(float *inp, float *filter, float *out,
                        int c1, int h1, int w1,
                        int n2, int h2, int w2)
{
    // inp的offset
    int offset1_c = h1 * w1, offset1_h = w1, offset1_w = 1;

    // filter的offset
    int c2 = c1;
    int offset2_n = c2 * h2 * w2, offset2_c = h2 * w2, offset2_h = w2, offset2_w = 1;

    // out的offset
    int h3 = h1 - h2 + 1;
    int w3 = w1 - w2 + 1;
    int offset3_c = h3 * w3, offset3_h = w3, offset3_w = 1;

    for (int i = n2 * h3 * w3 - 1; i >= 0; --i)
        out[i] = 0.;

    int jdx = 0;
    #pragma omp parallel for num_threads(8)
    for (int n = 0; n < n2; ++n)
        for (int i = 0; i < c2; ++i)
            for (int j = 0; j < h2; ++j)
                for (int k = 0; k < w2; ++k) {
                    float tmp = filter[jdx++];
                    int idx = 0;
                    for (int h = 0; h < h3; ++h)
                        for (int w = 0; w < w3; ++w)
                            out[n * offset3_c + (idx++)] += tmp * \
                                inp[i * offset1_c + (h + j) * offset1_h + (w + k) * (offset1_w)];
                }
}

void naive_convolution3(float *inp, float *filter, float *out,
                        int c1, int h1, int w1,
                        int n2, int h2, int w2)
{
    // inp的offset
    int offset1_c = h1 * w1, offset1_h = w1, offset1_w = 1;

    // filter的offset
    int c2 = c1;
    int offset2_n = c2 * h2 * w2, offset2_c = h2 * w2, offset2_h = w2, offset2_w = 1;

    // out的offset
    int h3 = h1 - h2 + 1;
    int w3 = w1 - w2 + 1;
    int offset3_c = h3 * w3, offset3_h = w3, offset3_w = 1;

    //*
    for (int i = n2 * h3 * w3 - 1; i >= 0; --i)
        out[i] = out[i] > 0. ? out[i] : 0.;
    //*/
}

long test_time(int c1, int h1, int w1, int n2, int h2, int w2)
{
    // 申请input的空间及对input初始化
    float *inp = new float[c1 * h1 * w1];
    for (int i = c1 * h1 * w1 - 1; i >= 0; --i)
        inp[i] = randf();

    // 申请卷积核的空间及对卷积核初始化
    int c2 = c1;
    float *filter = new float[n2 * c2 * h2 * w2];
    for (int i = n2 * c2 * h2 * w2 - 1; i >= 0; --i)
        filter[i] = randf();

    // 申请output的空间
    int c3 = n2;
    int h3 = h1 - h2 + 1;
    int w3 = w1 - w2 + 1;
    float *out = new float[c3 * h3 * w3];

    // 进行最原始的卷积
    auto start = system_time();
    naive_convolution2(inp, filter, out, c1, h1, w1, n2, h2, w2);
    auto finish = system_time();

    // 保存输入、卷积核、输出，用于python读取并检验
    /*
    write_array("input.array", (char *)inp, sizeof(float) * (c1 * h1 * w1));
    write_array("filter.array", (char *)filter, sizeof(float) * (n2 * c2 * h2 * w2));
    write_array("output.array", (char *)out, sizeof(float) * (c3 * h3 * w3));
    //*/

    delete []inp; 
    delete []filter;
    delete []out;

    return finish - start;
}

float test_accuracy(int c1, int h1, int w1, int n2, int h2, int w2)
{
    // 申请input的空间及对input初始化
    float *inp = new float[c1 * h1 * w1];
    for (int i = c1 * h1 * w1 - 1; i >= 0; --i)
        inp[i] = randf();

    // 申请卷积核的空间及对卷积核初始化
    int c2 = c1;
    float *filter = new float[n2 * c2 * h2 * w2];
    for (int i = n2 * c2 * h2 * w2 - 1; i >= 0; --i)
        filter[i] = randf();

    // 申请output的空间
    int c3 = n2;
    int h3 = h1 - h2 + 1;
    int w3 = w1 - w2 + 1;
    float *out1 = new float[c3 * h3 * w3];
    float *out2 = new float[c3 * h3 * w3];
    for (int i = c3 * h3 * w3 - 1; i >= 0; --i) {
        out1[i] = randf();
        out2[i] = randf();
    }

    naive_convolution(inp, filter, out1, c1, h1, w1, n2, h2, w2);
    naive_convolution2(inp, filter, out2, c1, h1, w1, n2, h2, w2);

    float err = 0.;
    for (int i = c3 * h3 * w3 - 1; i >= 0; --i)
        if (out1[i] > out2[i])
            err += (out1[i] - out2[i]);
        else
            err += (out2[i] - out1[i]);
    err /= (c3 * h3 * w3);

    delete []inp; 
    delete []filter;
    delete []out1;
    delete []out2;

    return err;
}

int main()
{
    seed();

    // 测试时间
    //*
    for (int i = 50; i < 500; i += 10) {
        auto t = test_time(3, i, i, 64, 7, 7);
        cout << t << ',';
    }
    cout << endl;
    //*/

    // 测试正确性
    /*
    for (int i = 50; i < 500; i += 10)
        cout << test_accuracy(3, i, i, 64, 7, 7) << ',';
    cout << endl;
    //*/

    return 0;
}
