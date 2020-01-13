#include "storage.h"

namespace snn
{

Storage& Storage::getInstance()
{
    static Storage instance(true);
    return instance;
}

Storage::Storage(bool _maintain): maintain(_maintain) {}

Storage::~Storage()
{
    clear();
}

int Storage::allocate(float *&data, int *&ref, int size)
{
    // ==>if：当Storage维护了buffer且存在空闲的内存块的时候，进入if分支
    // ==>else：当需要重新申请内存的时候，进入else分支
    if (maintain && unused.size() > 0)
    {
        // 在空闲的内存块中寻找刚好比申请的内存要大的内存块
        // 不一定能寻找到，如果所有的内存块都比申请的内存要小，it就会指向end()
        auto it = unused.begin();
        for (; it != unused.end(); ++it)
            if ((*it).first >= size)
                break;

        // ==>if：所有内存块都比申请的要小
        // ==>else：存在内存块比申请的大
        if (it == unused.end())
        {
            // 执行：
            // 1.找到最大的内存块
            // 2.让data和ref指向最大的内存块
            // 3.扩容

            // 让it指向最大的空闲内存块
            --it;

            // 让data和ref指向最大的内存块
            data = (*it).second;
            ref = pool[data].first;
            unused.erase(it);

            // 将data指向的内存扩大到size
            pool.erase(data);
            delete [] data;
            data = new float[size];
            pool.insert({data, std::make_pair(ref, size)});
        }
        else
        {
            // 因为set是有序的，所以此时it指向的一定所有大于申请大小的内存块中最小的那个
            // 为了效率，此时就不再重新申请刚好等于size的内存，通过浪费一点内存来取得效率
            data = (*it).second;
            ref = pool[data].first;
            unused.erase(it);
        }
    }
    else
    {
        data = new float[size];
        ref = new int;
        pool.insert({data, std::make_pair(ref, size)});
    }

    // 对新分配的内存，将其*ref设置为1,
    // 除了这个地方，在Storage的任何地方都不应该再操作*ref
    // 对*ref的操作应该交给使用内存的对象
    *ref = 1;

    return 0;
}

int Storage::free(float *data)
{
    if (pool.find(data) == pool.end())
        return -1; // data指向的内存不是由Storage申请的

    int *ref = pool[data].first;
    if (*(ref) != 0)
        return -2; // data指向的内存还被其它对象引用，不能释放

    // ==>if：如果Storage维护了一个buffer，进入if分支
    // ==>else：如果Storage只提供一个new和delete的wrapper，进入else分支
    if (maintain)
    {
        unused.insert(std::make_pair(pool[data].second, data));
    }
    else
    {
        pool.erase(data);
        delete [] data;
        delete ref;
    }

    return 0;
}

int Storage::clear()
{
    // 每个内存都必须没有被引用
    for (auto it = pool.begin(); it != pool.end(); ++it)
        if (*((it->second).first) != 0)
            return -1;

    for (auto it = pool.begin(); it != pool.end(); ++it)
    {
        // 删除data
        delete [] it->first;
        // 删除引用计数
        delete (it->second).first;
    }

    pool.clear();
    unused.clear();
}

std::vector<std::vector<int>> Storage::info()
{
    std::vector<std::vector<int>> ret(2);

    // pool内每个内存块的大小，乘以4是因为元素为float
    for (auto it = pool.begin(); it != pool.end(); ++it)
        ret[0].push_back((it->second).second * 4);

    // unused内每个内存块的大小，乘以4是因为元素为float
    for (auto it = unused.begin(); it != unused.end(); ++it)
        ret[1].push_back((*it).first * 4);

    return ret;
}

} // namespace snn
