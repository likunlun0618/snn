#include <fstream>
#include "io.h"

void write_array(std::string filename, float *data, std::vector<int> dim)
{
    std::ofstream fout;
    fout.open(filename, std::ofstream::binary);

    // 保存头部
    // 第0到3个字节表示有几个维度，假设为n
    // 第4到4n+4-1个字节表示每个维度的值
    int head[dim.size() + 1];
    head[0] = dim.size();
    for (int i = 1; i <= dim.size(); ++i)
        head[i] = dim[i - 1];
    fout.write((char *)head, sizeof(int) * (dim.size() + 1));

    // 保存实际数据
    int length = 1;
    for (int i = 1; i <= dim.size(); ++i)
        length *= head[i];
    fout.write((char *)data, sizeof(float) * length);

    fout.close();
}
