#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <net.h>
#include <tensor.h>

static std::vector<std::string> joint_names = {
    "right ankle", "right knee", "right hip",
    "left hip", "left knee", "left ankle",
    "pelvis", "thorax", "upper neck", "head top",
    "right wrist", "right elbow", "right shoulder",
    "left shoulder", "left elbow", "left wrist"
};

struct Joint
{
    int x, y;
    float p;
    Joint(int _x, int _y, float _p): x(_x), y(_y), p(_p) {}
};

std::vector<Joint> computeJoints(snn::Tensor heatmap)
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
        std::cout << "usage: " << argv[0] << " [input path] [output path] [model txt] [model array]" << std::endl;
        return 0;
    }

    // 根据传入的参数读取图片
    snn::Tensor img = readImage(argv[1]);
    std::vector<snn::Tensor> inp, out;
    inp.push_back(img);

    // 生成网络
    snn::Net hg(argv[3]);
    // 加载网络参数
    hg.load(argv[4]);

    // 前向传播
    long t1 = time();
    hg.forward(inp, out);
    long t2 = time();

    // 从heatmap中计算关节点坐标
    std::vector<Joint> joints = computeJoints(out[0]);
    for (int i = 0; i < joints.size(); ++i)
    {
        std::cout << joint_names[i] << ": (" << joints[i].x << "," << joints[i].y << "), p: " << joints[i].p << std::endl;
    }
    std::cout << "forward time: " << t2 - t1 << " us" << std::endl;

    // 绘制关节点
    cv::Mat src_img = cv::imread(argv[1]);
    for (int i = 0; i < joints.size(); ++i)
        if (joints[i].p > 0.5)
            circle(src_img, cv::Point(joints[i].x, joints[i].y), 3, cv::Scalar(0,255,0), -1);
    cv::imwrite(argv[2], src_img);

    return 0;
}
