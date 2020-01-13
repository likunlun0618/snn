#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensor.h>
#include <net.h>

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

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cout << "usage: " << argv[0] << " [model txt] [model array] [input image]" << std::endl;
        return 0;
    }

    // 创建模型
    snn::Net net(argv[1]);
    // 加载参数
    net.load(argv[2]);

    // 读取输入图片
    snn::Tensor img = readImage(argv[3]);

    vector<snn::Tensor> inp, out;
    inp.push_back(img);

    net.forward(inp, out);
    for (int i = 0; i < out[0].w; ++i)
        std::cout << out[0].data[i] << std::endl;

    return 0;
}
