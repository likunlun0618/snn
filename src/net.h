#ifndef NET_H
#define NET_H

#include <string>
#include <vector>
#include "module.h"

using std::string;
using std::vector;

namespace snn
{

/*
1.Node是表示计算图中的节点的结构。
2.成员说明：
    * p：表示该节点的类型
    * pre：存储该节点的前驱节点
    * next：存储该节点的后继节点
    * parameter：存储该节点在参数文件中的参数的索引
*/
class Node
{
public:
    Module *p;
    vector<int> pre;
    vector<int> next;
    vector<int> parameter;
    string name;

    // 将p设置为NULL
    Node();
    
    // 判断p是否为空，如果不为空，手动释放Module
    ~Node();

    void set(Module *_p, vector<int> _parameter, \
             vector<int> _pre, vector<int> _next, string _name);
};

/*
静态图得保证堆上的内存不会反复的申请释放，而是申请刚好足够的内存，一直维持到析构
*/
class Net
{
public:
    // 从计算图的配置文件中创建网络
    Net(string filename);

    ~Net();

    // 加载参数
    // 文件内容为几个Tensor的直接拼接（包含头部），可以读取出一个“Tensor”的“数组”
    // 在计算图的配置文件中需要指明参数在这个“数组”中的索引
    int load(string filename);

    // 计算
    // 计算图中所有没有前驱的节点，都是输入节点
    // 有多个输入时，它们的顺序应该和图中的节点编号从小到大对应
    // 计算图中所有没有后继的节点，都是输出节点
    // 有多个输出时，它们的顺序和图中节点编号从小到大对应
    int forward(const std::vector<Tensor> &inp, std::vector<Tensor> &out);

    void mergeConvBN();

//private:
public:
    // 保存计算图中的所有节点
    // 输入节点指向NULL
    // Module **modules;
    int num_modules;
    int num_input;
    int num_output;

    // 计算图
    // int表示索引，和modules对应
    // 当一个节点有多个前驱的时候，应该按从小到大的顺序排列
    // TODO：不按照节点编号排序，而是按照计算图的配置文件中的节点顺序
    // vector<vector<int>> pre_graph;
    // vector<vector<int>> next_graph;
    // 从节点到参数索引的映射
    // 依靠这个映射从二进制的参数文件中加载参数
    // vector<vector<int>> parameters;
    // 保存拓扑排序的结果，确保输入节点在最开始，输出节点在最后
    vector<int> order;
    vector<vector<int>> free_order;

    // 计算图的节点
    Node *node;

    // 保存中间变量
    // 因为要保存输入，所以buffer的大小是op数+num_input
    // 如果将buffer放到forward中声明，每次前向传播都会去堆上申请内存
    // 编译期不能确定buffer的长度，buffer的长度和计算图的节点数有关，所以不能使用array
    // 在加载计算图后，用reserve确定长度，避免内存碎片
    vector<Tensor> buffer;

    // 避免vector不停地申请释放
    // 在构造函数中也用reserve直接确定长度，即图中节点的数目
    vector<Tensor> inp_buffer;
    //vector<Tensor> out_buffer;

    // 将计算图的配置文件从便于人阅读的形式转换成便于代码处理的形式
    // 包含了合法性检查
    vector<string> simplify(string filename);

    // 从配置文件中的一行解析一个节点的信息
    // 返回错误代码
    int parse(string &line, int &id, vector<int> &pre, vector<int> &next, \
              vector<int> &parameter, string &name, string &options);

    // 拓扑排序，对order进行赋值
    // 返回-1表示有环
    int topologicalSort();
    // 拓扑排序的辅助函数
    // visited中0表示white，1表示gray，2表示black
    bool visit(int index, vector<int> &visited);
};

} // namespace snn

#endif // NET_H
