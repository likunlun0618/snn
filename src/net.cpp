#include <sys/timeb.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include "net.h"
#include "parse.h"
#include "storage.h"
#include "global.h"

static long system_time()
{
    timeb t;
    ftime(&t);
    return t.time * 1000 + t.millitm;
}

Node::Node(): p(NULL) {}

Node::~Node()
{
    if (p)
        delete p;
}

void Node::set(Module *_p, vector<int> _parameter, \
               vector<int> _pre, vector<int> _next)
{
    p = _p;
    parameter = _parameter;
    pre = _pre;
    next = _next;
}

// Net的构造函数里的处理步骤
// 原始文件 ==> 标准文件 ==> 解析数据 ==> 给参数赋值
Net::Net(string filename): num_input(0), num_output(0), node(NULL)
{
    vector<string> lines = simplify(filename);
    // 假设lines此时已经进行了合法性检查

    // 初始化
    // 注意每一个输入的节点和输出的节点也都算作一个Module
    // 输入节点和输出节点一定在计算图的起始和结束
    // 输入节点的顺序和输出节点的顺序决定了forward传入和传出的vector中的顺序
    num_modules = lines.size();
    node = new Node[num_modules];

    // 给3个buffer申请空间
    buffer.resize(num_modules);
    inp_buffer.reserve(num_modules);
    //out_buffer.reserve(num_modules);

    // 逐行处理
    for (string &line : lines)
    {
        int id;
        vector<int> parameter, pre, next;
        string name, options;

        // 如果括号里没有东西，vector的话size为0，string为""
        // 没有对应的选项同理
        // 因为在simplify中做过输入合法性检测，所以此处不再检查错误代码
        parse(line, id, pre, next, parameter, name, options);

        Module *p = NULL;
        if (name != "input" && name != "output")
            p = ModuleFactory::getInstance().create(name, options);
        else
            name == "input" ? ++num_input : ++num_output;

        node[id].set(p, parameter, pre, next);
    }

    // 拓扑排序，找到节点的计算依赖关系
    topologicalSort();
}

int Net::load(string filename)
{
    std::ifstream fin(filename, std::ifstream::binary);

    // 读取Tensor的个数
    int num;
    fin.read((char *)&num, sizeof(int));

    // 读取num个Tensor的数据
    float **p = new float*[num];
    int *size = new int[num];
    for (int i = 0; i < num; ++i)
    {
        int head[5];
        fin.read((char *)head, sizeof(int) * 5);
        size[i] = head[1] * head[2] * head[3] * head[4];
        p[i] = new float[size[i]];
        fin.read((char *)p[i], sizeof(float) * size[i]);
    }

    fin.close();

    // 读取每个节点的参数索引并load
    for (int i = 0; i < num_modules; ++i)
    {
        for (int j = 0; j < node[i].parameter.size(); ++j)
        {
            int para_id = node[i].parameter[j];
            node[i].p->load(p[para_id], size[para_id], j);
        }
    }

    for (int i = 0; i < num; ++i)
        delete [] p[i];
    delete [] p;
    delete [] size;
}

Net::~Net()
{
    if (node)
        delete [] node;
}

// 将配置文件从便于人阅读的形式转换成标准模式
vector<string> Net::simplify(string filename)
{
    // 从文件中读取原始的行
    vector<string> lines = parse::readFile(filename);
    // 删除注释
    lines = parse::deleteComments(lines);
    // 删除空行
    lines = parse::deleteEmptyLines(lines);
    return lines;
}

int Net::parse(string &line, int &id, vector<int> &pre, vector<int> &next,
               vector<int> &parameter, string &name, string &options)
{
    // 用空格做split
    vector<string> items = parse::split(line, " ");
    // id
    std::stringstream(items[0]) >> id;
    // pre
    items[1] = parse::parseItem(items[1], "pre");
    pre = parse::readArray(items[1], ",");
    // next
    items[2] = parse::parseItem(items[2], "next");
    next = parse::readArray(items[2], ",");
    // name
    name = parse::parseItem(items[3], "name");
    // options
    if (items.size() > 4)
        options = parse::parseItem(items[4], "options");
    // parameter
    if (items.size() > 5)
    {
        items[5] = parse::parseItem(items[5], "parameters");
        parameter = parse::readArray(items[5], ",");
    }

    return 0;
}

int Net::topologicalSort()
{
    // 拓扑排序
    vector<int> mark(num_modules, 0);
    for (int i = 0; i < num_modules; ++i)
        if (visit(i, mark))
            return -1; // 有环

    // 经历过拓扑排序之后，order中应该已经排好了数据
    // 根据order计算释放节点的顺序
    free_order.resize(num_modules);
    vector<bool> visited(num_modules, false);
    for (int i = 0; i < num_input; ++i)
        visited[i] = true;
    for (int i = num_input; i < num_modules - num_output; ++i)
    {
        visited[i] = true;
        for (int j : node[i].pre)
        {
            bool invalid = true;
            for (int k : node[j].next)
                if (!visited[k])
                {
                    invalid = false;
                    break;
                }
            if (invalid)
                free_order[i].push_back(j);
        }
    }

    return 0;
}

bool Net::visit(int index, vector<int> &visited)
{
    if (visited[index] == 0)
    {
        visited[index] = 1;
        // 最开始遍历输入节点的时候，不会进入这个循环，因为输入没有前驱
        // 所以输入一定会按照配置文件中的顺序最先出现在order中
        // 而最后遍历到第一个输出节点的时候，所有输出节点的前驱一定已经被遍历了
        // 所以在循环中每次进入visit都是直接返回false
        // 所以输出节点一定会按照配置文件中的顺序出现在order中
        for (int i : node[index].pre)
            if (visit(i, visited))
                return true;
        visited[index] = 2;
        order.push_back(index);
        return false;
    }
    else if (visited[index] == 1)
        return true;
    else
        return false;
}

int Net::forward(const vector<Tensor> &inp, vector<Tensor> &out)
{
    long t1, t2;
    
    // buffer, inp, out在每次运行完之后理应自动恢复到干净状态
    // 后面注释掉这4行代码观察结果是否会错误
    for (int i = 0; i < num_modules; ++i)
        buffer[i].clear();
    inp_buffer.clear();
    //out_buffer.clear();

    // 将inp写到buffer中
    // 默认buffer已经是干净状态
    for (int i = 0; i < num_input; ++i)
        buffer[i] = inp[i];

    // auto info = Storage::getInstance().info();
    // std::cout << info[0].size() << std::endl;
    // 11

    // 依次计算
    // 默认inp_buffer和out_buffer是干净状态
    for (int i = num_input; i < order.size() - num_output; ++i)
    {

        t1 = system_time();

        int module_id = order[i];

        global_module = module_id;

        // 寻找当前节点的前驱节点
        for (int &pre_id : node[module_id].pre)
            inp_buffer.push_back(buffer[pre_id]);

        // 因为Tensor的大小是固定的，且不管理内存
        // inp_buffer和out_buffer又都是已经分配好的空间
        // 所以不论运行几次，都不会发生反复的堆内存申请和释放
        // TODO：检查p是否为空
        // std::cout << module_id << std::endl;
        // t1 = system_time();
        buffer[module_id] = node[module_id].p->forward(inp_buffer);
        // t2 = system_time();
        /* 这段代码应该是不符合逻辑的，不需要放到next的buffer中去
        // 将输出放到对应的buffer中
        for (int i = 0; i < out_buffer.size(); ++i)
            buffer[node[module_id].next[i]] = out[i];
        */

        // 将无用的前驱节点释放掉
        // 减小内存占用
        // 这个过程是固定的，可以提前做好
        for (int free_id : free_order[module_id])
            buffer[free_id].clear();

        // 恢复到干净状态
        inp_buffer.clear();
        //out_buffer.clear();

        // auto info = Storage::getInstance().info();
        // std::cout << i << ":" << info[0].size() << "," << info[1].size() << std::endl;

        t2 = system_time();
        /*
        if (global == 1 && global_module == 197)
        {
            std::cout << "test!" << std::endl;
            std::cout << t2 - t1 << std::endl;
            while (true);
        }
        //*/
    }

    for (int i = num_modules - num_output; i < num_modules; ++i)
    {
        // 此时order[i]应该等于i
        int pre_id = node[i].pre[0];
        out.push_back(buffer[pre_id]);
        // 恢复到干净状态
        buffer[pre_id].clear();
    }

    // auto info = Storage::getInstance().info();
    // std::cout << "end:" << info[0].size() << "," << info[1].size() << std::endl;

    // 正常工作
    return 0;
}

/*
void Net::mergeConvBN()
{
    // 找到每一组满足融合条件的conv->bn
    // 将bn的next添加到conv中，将bn的next里的每一个元素的pre里的bn改成conv
    // 将bn的参数融进conv
    // 将bn的pre和next都置空

    // 重新计算拓扑排序
    // 当一个节点既没有pre也没有next的时候，就跳过它

    // 修改代码，因为此时拓扑排序的结果的长度可能会小于num_modules
}
*/
