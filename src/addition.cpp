#include <iostream>
#include "addition.h"

Module* createAddition(std::string options)
{
    Addition *add = new Addition;
    return (Module*)add;
}

Tensor Addition::forward(const std::vector<Tensor> &inp)
{
    int num = inp.size();
    if (num < 2)
    {
        std::cout << "Addition error: tensor < 2" << std::endl;
        exit(-1);
    }

    int c = inp[0].c;
    int h = inp[0].h;
    int w = inp[0].w;
    
    for (int i = 1; i < num; ++i)
        if (inp[i].c != c || inp[i].h != h || inp[i].w != w)
        {
            std::cout << "Addition error: dimensions do not match" << std::endl;
            exit(-1);
        }

    Tensor out(1, c, h, w);

    int s = h * w;
    #pragma omp parallel for num_threads(4)
    for (int ci = 0; ci < c; ++ci)
    {
        float *input = inp[0].data + ci * s;
        float *output = out.data + ci * s;
        for (int i = 0; i < s; ++i)
            output[i] = input[i];
        for (int j = 1; j < num; ++j)
        {
            input = inp[j].data + ci * s;
            for (int i = 0; i < s; ++i)
                output[i] += input[i];
        }
    }

    return out;
}

int Addition::parameters() const
{
    return 0;
}
