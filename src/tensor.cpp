#include <iostream>
using std::cout;
using std::endl;
#include <sys/timeb.h>
#include <fstream>
#include "tensor.h"
#include "storage.h"
#include "global.h"

static long system_time()
{
    timeb t;
    ftime(&t);
    return t.time * 1000 + t.millitm;
}

Tensor::Tensor(int _c, int _h, int _w): data(NULL), ref(NULL)
{
    resize(1, _c, _h, _w);
}

Tensor::Tensor(int _n, int _c, int _h, int _w): data(NULL), ref(NULL)
{
    resize(_n, _c, _h, _w);
}

Tensor::Tensor(const Tensor &t): data(NULL), ref(NULL)
{
    *this = t;
}

// 浅拷贝
// 注意引用数的变化
Tensor& Tensor::operator=(const Tensor &t)
{
    // 判断是否为自身，如果是的话，直接返回
    // 猜想：此处也可以用t != *this来判断，因为这样判断是判断每个成员是否相等
    // 当两个Tensor的所有成员都相等的时候，说明*ref为n+2，且赋值之后，*ref还是n+2
    // （其中n是除t和*this之外的指向t.data的Tensor的个数）
    if (&t != this)
    {
        // 将*this置为空Tensor
        clear();

        data = t.data;
        ref = t.ref;
        n = t.n;
        c = t.c;
        h = t.h;
        w = t.w;

        // 如果t不为空Tensor，赋值之后就会又多一个Tensor指向t.data，所以引用计数要+1
        if (ref)
            ++(*ref);
    }
    return *this;
}

Tensor::~Tensor()
{
    clear();
}

void Tensor::clear()
{
    if (data)
    {
        // 此处默认了当data不为NULL的时候，ref也必然不为NULL
        // 如果不满足这个条件，说明一定有某个地方出现了data和ref没有同步修改的问题
        --(*ref);
        if (*ref == 0)
            Storage::getInstance().free(data);
    }
    data = NULL;
    ref = NULL;
    n = 0;
    c = 0;
    h = 0;
    w = 0;
}

int Tensor::resize(int _c, int _h, int _w)
{
    return resize(1, _c, _h, _w);
}

// 注意即使resize申请的内存大小和*this本来持有的内存相等，也要向Storage重新申请
// resize就是Tensor向Storage申请的接口
int Tensor::resize(int _n, int _c, int _h, int _w)
{
    // long t1, t2;
    // t1 = system_time();

    int size = _n * _c * _h * _w;
    if (size > 0)
    {
        clear();
        // ref会自动等于1，如果不等于1，必然出现了问题
        // 除了这个地方以外，*ref的所有变化都是在Tensor里进行的，Storage不会操作*ref;
        Storage::getInstance().allocate(data, ref, size);
        n = _n;
        c = _c;
        h = _h;
        w = _w;    
    }
    else
    {
        // 若申请的size<=0，则不会对*this进行任何改变，直接返回错误代码
        return -1;
    }

    // t2 = system_time();

    /* 用均匀分布给data赋值，便于debug
    for (int i = 0; i < size; ++i)
        data[i] = (float)rand() / (float)RAND_MAX;
    //*/

    /*
    if (global == 1 && global_module == 1)
    {
        std::cout << "rand?" << std::endl;
        std::cout << t2 - t1 << std::endl;
        while (true);
    }
    */

    return 0;
}

void Tensor::save(std::string filename)
{
    std::ofstream fout;
    fout.open(filename, std::ofstream::binary);

    // 保存维度信息
    // 第一个4表示Tensor的总维度是4
    // 后面4个数字依次是对应的维度
    int head[5] = {4, n, c, h, w};
    fout.write((char *)head, sizeof(int) * 5);

    // 保存数值
    int length = n * c * h * w;
    fout.write((char *)data, sizeof(float) * length);

    fout.close();
}

int Tensor::load(std::string filename)
{
    std::ifstream fin(filename, std::ifstream::binary);

    // 读取Tensor的个数，应该为1
    int num;
    fin.read((char *)&num, sizeof(int));

    // 读取Tensor的头
    int head[5];
    fin.read((char *)head, sizeof(int) * 5);
    int s = 1;
    for (int i = 1; i < 5; ++i)
        s *= head[i];

    // 读取float的数据
    float *tmp = new float[s];
    fin.read((char *)tmp, sizeof(float) * s);
    fin.close();

    load(tmp);
    delete [] tmp;

    return 0;
}

void Tensor::load(float *_data)
{
    for (int i = n * c * h * w - 1; i >= 0; --i)
        data[i] = _data[i];
}
