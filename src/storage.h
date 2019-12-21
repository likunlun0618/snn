#ifndef STORAGE_H
#define STORAGE_H

#include <set>
#include <vector>
#include <unordered_map>

/*
1.以Tensor为单元管理内存
2.只管理Tensor的内存
3.Tensor的内存只能使用Storage管理
4.Storage只对Tensor可见，在任何其它地方都不应该被使用
*/
class Storage
{
public:
    // 如果采用单例模式的话，假如网络提前释放，Storage申请的内存却不会释放
    // 但如果不采用单例模式（而是作为网络的成员），那么想在网络外部定义Tensor就还得同时定义
    // Storage，既麻烦，又增加了系统的复杂性
    // ==> 可以添加一个手动释放内存的函数来暂时解决这个问题
    static Storage& getInstance();

    // 分配指定大小的内存
    // 如果有空闲的内存且大小比需求的大，那么就浪费一点空间换取不频繁的申请释放内存
    // 如果有空闲的内存且大小比需求的小，那么释放该内存，重新申请更大的空间
    // 原则是保证内存的“块”数最少
    // 返回值为错误代码
    int allocate(float *&data, int *&ref, int size);

    // 释放内存，将该内存重新加入到空闲的内存池里去
    // 如果*ref不为空，那么将不能释放内存
    // 返回值为错误代码
    int free(float *data);

    // 手动释放Storage的所有内存
    // 只有当所有的*ref都为0的时候才能成功
    int clear();

    // 返回pool和unused的内存信息
    // ret[0]表示pool内每个内存块的大小
    // ret[1]表示unused内每个内存块的大小
    std::vector<std::vector<int>> info();

private:
    // 参数为指定Storage是否要维护buffer
    // 如果为true，表示Storage会一直维护申请的内存
    // 如果为false，表示Storage会直接用new和delete来分配内存
    Storage(bool _maintain);

    // 释放所有内存
    // 再次注意这里只维护所有tensor的内存
    ~Storage();

    // 是否维护一个buffer
    bool maintain;

    // 保存所有已经申请的内存块的信息
    // 内存块的首地址 ==> (引用计数的地址, 内存块的长度)
    std::unordered_map<float*, std::pair<int*, int>> pool;

    // 在maintain=true的时候才有用
    // 保存所有空闲的内存块的信息，即(内存块的长度，内存块的首地址)
    // pair的大小比较为先比较第一个元素，如果相同再比较第二个，所以长度必须放在第一个
    std::set<std::pair<int, float*>> unused;
};

#endif // STORAGE_H
