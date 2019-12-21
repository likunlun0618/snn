#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "net.h"
#include "tensor.h"
#include "image.h"
#include "global.h"

int global = 0;
int global_module = 0;

static std::vector<std::string> joint_names = {
        "right ankle", "right knee", "right hip",
        "left hip", "left knee", "left ankle",
        "pelvis", "thorax", "upper neck", "head top",
        "right wrist", "right elbow", "right shoulder",
        "left shoulder", "left elbow", "left wrist"
};

class Joint
{
public:
    int x;
    int y;
    float p; // 概率
    Joint(int _x, int _y, float _p): x(_x), y(_y), p(_p) {}
};

std::vector<Joint> computeJoints(Tensor heatmap)
{
    int c = heatmap.c;
    int h = heatmap.h;
    int w = heatmap.w;

    std::vector<Joint> joints;
    for (int i = 0; i < c; ++i)
    {
        int x, y;
        float p;
        for (int j = 0; j < h; ++j)
            for (int k = 0; k < w; ++k)
            {
                // x，y，p的初值
                if (j == 0 && k == 0)
                {
                    x = k;
                    y = j;
                    p = heatmap.data[i * h * w + j * w + k];
                }

                float tmp = heatmap.data[i * h * w + j * w + k];
                if (tmp > p)
                {
                    x = k;
                    y = j;
                    p = tmp;
                }
            }
        joints.push_back(Joint(x * 4, y * 4, p));
    }

    return joints;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "usage: " << argv[0] << " [input path] [output path] [model txt] [model array]" << std::endl;
        return 0;
    }

    // 根据传入的参数读取图片
    Tensor img = imread(argv[1]);
    std::vector<Tensor> inp, out;
    inp.push_back(img);

    // 构造网络
    Net hg(argv[3]);
    hg.load(argv[4]);

    // 前向传播
    hg.forward(inp, out);

    // 从heatmap中计算关节点坐标
    std::vector<Joint> joints = computeJoints(out[0]);
    for (int i = 0; i < joints.size(); ++i)
    {
        std::cout << joint_names[i] << ": (" << joints[i].x << "," << joints[i].y << "), p: " << joints[i].p << std::endl;
    }

    cv::Mat src_img = cv::imread(argv[1]);
    for (int i = 0; i < joints.size(); ++i)
    {
        if (joints[i].p > 0.5)
        {
            circle(src_img, cv::Point(joints[i].x, joints[i].y), 3, cv::Scalar(0,255,0), -1);
        }
    }
    cv::imwrite(argv[2], src_img);

    return 0;
}
