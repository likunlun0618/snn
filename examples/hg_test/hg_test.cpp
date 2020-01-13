#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <storage.h>
#include <net.h>
//#include "time.h"
//#include "global.h"

//int global = 0;
//int global_module = 0;

//long var1 = 0;

using namespace std;

long time()
{
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec * 1e6 + t.tv_usec;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "usage: " << argv[0] << " [model txt] [model array] [input array] [output array]" << std::endl;
        exit(0);
    }

    // 构建网络模型
    Net net(argv[1]);
    net.load(argv[2]);

    // 合并可以合并的卷积和BN
    // net.mergeConvBN();

    // 准备输入和输出
    Tensor img(1, 3, 256, 256);
    img.load(argv[3]);
    vector<Tensor> inp, out;
    inp.push_back(img);

    /*
    std::cout << "start loop ..." << std::endl;
    while (true)
    {
        out.clear();
        net.forward(inp, out);
    }
    //*/

    // 先运行几次，申请足够的缓存
    //*
    net.forward(inp, out);
    out.clear();
    net.forward(inp, out);
    out.clear();
    net.forward(inp, out);
    out.clear();
    //*/
    //global = 1;
    // 前向传播
    long t1, t2;
    t1 = time();
    net.forward(inp, out);
    t2 = time();

    // 和pytorch版本的结果做对比
    Tensor output(1, out[0].c, out[0].h, out[0].w);
    output.load(argv[4]);
    float s = 0.;
    for (int i = 0; i < output.c * output.h * output.w; ++i)
        s += abs(output.data[i] - out[0].data[i]);
    s /= (output.c * output.h * output.w);
    cout << "average error:" << s << endl;

    // 打印时间
    cout << "time:" << t2 - t1 << "us" << endl;

    // 打印storage的信息
    cout << "storage info:" << endl;
    auto info = Storage::getInstance().info();
    int total_size = 0;
    for (int item : info[0])
        total_size += item;
    cout << "total size:" << info[0].size() << ", " << (float)total_size / (1024*1024) << "MB" << endl;
    int empty_size = 0;
    for (int item : info[1])
        empty_size += item;
    cout << "empty size:" << info[1].size() << ", " << (float)empty_size / (1024*1024) << "MB" << endl;

    // while (true);

    return 0;
}
