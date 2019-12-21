#include <sstream>
#include "reshape.h"

Module* createReshape(std::string options)
{
    int count = 0;
    for (char &ch : options)
        if (ch == ',')
        {
            ++count;
            ch = ' ';
        }

    int n, c, h, w;
    n = c = h = w = 1;
    if (count == 0)
        std::stringstream(options) >> w;
    else if (count == 1)
        std::stringstream(options) >> h >> w;
    else if (count == 2)
        std::stringstream(options) >> c >> h >> w;
    else
        std::stringstream(options) >> n >> c >> h >> w;

    Reshape *reshape = new Reshape(n, c, h, w);
    return (Module*)reshape;
}

Reshape::Reshape(int _n, int _c, int _h, int _w): n(_n), c(_c), h(_h), w(_w) {}

Tensor Reshape::forward(const std::vector<Tensor> &inp)
{
    Tensor out(inp[0]);

    // TODO：添加维度检测
    out.n = n;
    out.c = c;
    out.h = h;
    out.w = w;

    return out;
}

int Reshape::parameters() const
{
    return 0;
}

//int Reshape::load(float *data, int size, int index) {}
