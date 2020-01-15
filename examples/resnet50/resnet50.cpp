#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <net.h>

long time()
{
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec * 1e6 + t.tv_usec;
}

snn::Tensor readImage(std::string file)
{
    cv::Mat img = cv::imread(file, cv::IMREAD_UNCHANGED);
    if (!img.data)
    {
        std::cout << "open " << file << " failed." << std::endl;
        exit(0);
    }

    int c = img.channels() > 1 ? 3 : 1;
    int h = img.rows;
    int w = img.cols;
    snn::Tensor ret(1, c, h, w);

    int index = 0;
    int img_c = img.channels();
    for (int j = 0; j < c; ++j)
        for (int i = 0; i < h * w; ++i)
            ret.data[index++] = (float)img.data[i * img_c + j] / 255.;

    return ret;
}

// 读取ImageNet的label
std::vector<std::string> readLabels(std::string filename)
{
    std::vector<std::string> ret;
    
    char buffer[1024];
    std::ifstream file(filename);
    while (file.getline(buffer, sizeof(buffer)))
        ret.push_back(buffer);

    return ret;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "usage: " << argv[0] << " [image path] [model txt] [model array] [label file]" << std::endl;
        return 0;
    }

    // 构建模型
    snn::Net resnet(argv[2]);
    resnet.load(argv[3]);

    // 读取图片
    snn::Tensor img = readImage(argv[1]);
    std::vector<snn::Tensor> inp, out;
    inp.push_back(img);

    // 前向传播
    long t1 = time();
    resnet.forward(inp, out);
    long t2 = time();

    // 读取ImageNet的label
    std::vector<std::string> labels = readLabels(argv[4]);

    // 计算概率最大的类别
    int id = 0;
    for (int i = 0; i < out[0].w; ++i)
        if (out[0].data[i] > out[0].data[id])
            id = i;

    // softmax求概率
    float sum = 0.;
    for (int i = 0; i < out[0].w; ++i)
        sum += exp(out[0].data[i]);
    float p = exp(out[0].data[id]) / sum;

    // 输出结果
    std::cout << "forward time: " << t2 - t1 << " us" << std::endl;
    std::cout << "class id: " << id << std::endl;
    std::cout << "class name: " << labels[id] << std::endl;
    std::cout << "probability: " << p << std::endl;

    return 0;
}
