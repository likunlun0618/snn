#ifndef TENSOR_H
#define TENSOR_H

#include <string>

/*
1.Tensor不负责管理内存，用Storage自动管理内存
2.Tensor的大小是固定的，即两个指针加4个int
*/
class Tensor
{
public:
    // TODO：让data不暴露出来
    // 如果不暴露出来，没办法调用openBLAS的矩阵运算接口
    // 如果暴露出来，整个类就非常不安全
    float *data;

    // 引用计数
    // 使用指针才能保证所有指向同一块内存的Tensor有相同的引用计数
    // ref指向的那块空间应该被Storage管理
    // 为什么不使用shared_ptr：
    // 因为shared_ptr会自动使用delete，而我需要内存不被释放，一直存在于Storage中
    int *ref;

    // 关于存储维度的内存由谁来管理的思考：
    // 1.如果使用vector<int>表示维度，Tensor的复制，传递，返回都会在堆上发生申请和释放
    // 2.如果使用Storage来管理的话，复杂度上升太多
    // 3.考虑到这个框架的目标，使用固定的4个维度已经足够了
    int n, c, h, w;

    // 构建空Tensor
    Tensor();

    // 向Storage申请指定大小的空间，但是不初始化
    Tensor(int _c, int _h, int _w);
    Tensor(int _n, int _c, int _h, int _w);

    // 申请指定大小的空间，并使用_data的内容初始化
    // 注意：
    // 1.不是让data指向了_data的空间
    // 2.这里的申请不是用new申请，而是向Storage申请
    // Tensor(float *_data, int _c, int _h ,int _w);
    // Tensor(float *_data, int _n, int _c, int _h ,int _w);

    // 只做浅拷贝，但是要将ref指向的int加1
    // 实际上每次拷贝只拷贝了两个指针和4个int
    Tensor(const Tensor &t);

    // 移动构造函数
    // 复制所有成员，并将输入的Tensor置为空（data和ref为NULL，维度全为0）
    // *ref既没有增加，也没有减少
    // Tensor(Tensor &&t);

    // 和拷贝构造函数及移动拷贝构造函数功能相同
    Tensor& operator=(const Tensor &t);
    // Tensor& operator=(Tensor &&t);

    // 先判断data是否为NULL，如果是的话，就表明Tensor已经为空
    // 再将ref指向的int减1
    // 判断ref指向的int是否为0，如果是的话，通知Storage回收内存
    ~Tensor();

    // 手动释放
    // 流程和析构函数类似：判断data ==> 减少*ref ==> 判断*ref ==> 通知Storage回收内存
    void clear();

    // 只改变维度，不改变data和ref
    // void reshape(int _c, int _h, int _w);
    // void reshape(int _n, int _c, int _h, int _w);

    // 构造函数的手动版本
    // 先调用clear，再执行和构造函数一样的功能
    int resize(int _c, int _h, int _w);
    int resize(int _n, int _c, int _h, int _w);
    // int resize(float *_data, int _c, int _h, int _w);
    // int resize(float *_data, int _n, int _c, int _h, int _w);

    // 将Tensor的维度和数据存成文件
    // 这个函数在整个目标中其实是没用的，因为这只是一个前向框架，参数需要从别的地方读取
    // 但是可以用于debug
    void save(std::string filename);

    // 从文件中加载Tensor
    // 如果Tensor自己为空，那么用从文件中读取到的维度调用resize，再读取数据
    // 如果从文件里读取到的维度和Tensor自己的维度不匹配，就返回错误代码
    int load(std::string filename);

    // 复制数据，注意不是让data指向_data指向的空间，而是复制每一个元素
    void load(float *_data);
};

inline Tensor::Tensor(): data(NULL), ref(NULL), n(0), c(0), h(0), w(0) {}

#endif // TENSOR_H
