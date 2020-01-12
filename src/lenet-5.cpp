#include <iostream>
#include <sstream>
#include <vector>
#include "tensor.h"
#include "net.h"
#include "storage.h"
#include "image.h"
#include "time.h"
#include "global.h"

int global = 0;
int global_module = 0;

using namespace std;

void printVector(vector<int> &v)
{
    for (int i = 0; i < v.size(); ++i)
        if (i != v.size() - 1)
            cout << v[i] << ",";
        else
            cout << v[i];
    cout << endl;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cout << "usage: " << argv[0] << " [model txt] [model array] [input image]" << std::endl;
        return 0;
    }

    Net net(argv[1]);
    net.load(argv[2]);

    Tensor img = imread(argv[3]);

    vector<Tensor> inp, out;
    inp.push_back(img);

    // auto info = Storage::getInstance().info();
    // cout << info[0].size() << endl;
    // 11

    net.forward(inp, out);
    for (int i = 0; i < out[0].w; ++i)
        cout << out[0].data[i] << endl;

    return 0;
}
