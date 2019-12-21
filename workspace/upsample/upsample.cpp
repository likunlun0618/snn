#include <iostream>
#include <sys/timeb.h>
#include "random.h"
#include "io.h"

using namespace std;

long system_time()
{
    timeb t;
    ftime(&t);
    return t.time * 1000 + t.millitm;
}

void upsample(float *inp, float *out, int c, int h, int w, int s)
{
    int outh = h * s;
    int outw = w * s;
    # pragma omp parallel for num_threads(4)
    for (int ci = 0; ci < c; ++ci)
    {
        float *input = inp + ci * h * w;
        float *output = out + ci * outh * outw;
        int n = 0;
        for (int hi = 0; hi < outh; ++hi)
            for (int wi = 0; wi < outw; ++wi)
            {
                output[n++] = input[hi / s * w + wi / s];
            }
    }
}

int main()
{
    seed();

    int c = 128, h = 128, w = 128, s = 2;

    float *inp = new float[c * h * w];
    for (int i = c * h * w - 1; i >= 0; --i)
        inp[i] = randf();

    int outh = h * s;
    int outw = w * s;
    float *out = new float[c * outh * outw];

    long t1 = system_time();
    upsample(inp, out, c, h, w, s);
    long t2 = system_time();
    cout << "cpp time: " << t2 - t1 << " ms" << endl;

    write_array("inp.array", inp, {c, h, w});
    write_array("out.array", out, {c, outh, outw});

    return 0;
}
