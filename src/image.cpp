#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "tensor.h"

/*
1.对opencv的imread的封装，读取图片并返回Tensor
2.Tensor的每个元素在[0,1]之间，对应图片像素的[0,255]
3.图片的通道以BGR的顺序存储
4.将数据从opencv的HWC转成CHW
*/
Tensor imread(std::string filename)
{
    cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
    // 读取图片失败
    if (!img.data)
    {
        std::cout << "open " << filename << " failed." << std::endl;
        exit(0);
    }

    int c = img.channels() > 1 ? 3 : 1;
    int h = img.rows;
    int w = img.cols;
    Tensor ret(1, c, h, w);

    int index = 0;
    int img_c = img.channels();
    for (int j = 0; j < c; ++j)
        for (int i = 0; i < h * w; ++i)
            ret.data[index++] = (float)img.data[i * img_c + j] / 255.;

    return ret;
}

/*
1.对opencv的imwrite的封装，将Tensor存成图片
2.Tensor的值应该在[0,1]之间，如果在这个范围之外，会进行截断
*/
int imwrite(std::string filename, const Tensor &inp)
{
    int c = inp.c;
    int h = inp.h;
    int w = inp.w;

    cv::Mat img;
    if (c == 1)
        img.create(h, w, CV_8UC1);
    else if (c == 3)
        img.create(h, w, CV_8UC3);
    else
    {
        std::cout << "channels of tensor should be 1 or 3." << std::endl;
        exit(0);
    }

    int index = 0;
    for (int j = 0; j < c; ++j)
        for (int i = 0; i < h * w; ++i)
        {
            float val = inp.data[index++];
            unsigned char pixel;
            if (val > 1.)
                pixel = 255;
            else if (val < 0.)
                pixel = 0;
            else
                pixel = (unsigned char)(val * 255.);
            img.data[i * c + j] = pixel;
        }
    
    cv::imwrite(filename, img);

    return 0;
}
