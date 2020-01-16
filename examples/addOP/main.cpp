#include <iostream>
#include <vector>
#include <net.h>
#include <module.h>
#include "pow.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "usage: " << argv[0] << " [number] [pow.txt path]" << std::endl;
        return 0;
    }

    // 注册自定义OP
    snn::ModuleFactory::getInstance().registerModule("pow", createPow);

    snn::Net net(argv[2]);

    // 从argv中读取底数
    snn::Tensor base(1, 1, 1, 1);
    base.data[0] = std::stof(argv[1]);

    std::vector<snn::Tensor> inp, out;
    inp.push_back(base);

    net.forward(inp, out);

    std::cout << argv[1] << "^2 = " << out[0].data[0] << std::endl;

    return 0;
}
