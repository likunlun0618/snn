#include <iostream>
#include <vector>
#include <cmath>
#include <net.h>
#include <tensor.h>

using namespace std;

int main()
{
    snn::Net net("../resnet50/data/resnet50.txt");
    net.load("../resnet50/data/resnet50.array");

    snn::Tensor img(1, 3, 224, 224);
    img.load("data/inp.array");
    vector<snn::Tensor> inp, out;
    inp.push_back(img);

    net.forward(inp, out);

    snn::Tensor out2(1, out[0].c, out[0].h , out[0].w);
    out2.load("data/out.array");
    float s = 0.;
    for (int i = 0; i < out2.c * out2.h * out2.w; ++i)
        s += abs(out2.data[i] - out[0].data[i]);
    s /= (out2.c * out2.h * out2.w);
    cout << "average error: " << s << endl;

    float *p = out[0].data;
    int idx = 0;
    for (int i = 1; i < 1000; ++i)
        if (p[i] > p[idx])
            idx = i;
    cout << "max p: " << p[idx] << endl;
    cout << "class(0-indexed): " << idx << endl;


    return 0;
}
