#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <string>
#include <unordered_map>
#include "tensor.h"

namespace snn
{

class Module
{
public:
    // 注意Tensor是固定大小的，即两个指针加4个int

    // 最核心的接口
    // 通过优化，让静态图避免vector的不停地申请释放
    // 动态图只需要达到能用即可
    virtual Tensor forward(const std::vector<Tensor> &inp) = 0;

    // 返回参数的个数
    // 不是必需的组件
    virtual int parameters() const = 0;

    // 加载参数
    // 注意是拷贝_data的内容到参数而不是让参数指向_data，参数的内存也是由Storage管理的
    // 返回值为错误代码
    virtual int load(float *data, int size, int index);
    // 这个接口在使用静态图的时候不会使用，只是方便进行单元测试
    // 返回值为错误代码，例如文件的维度和参数的维度不匹配，就不加载参数，而是返回错误代码
    // virtual int load(std::string filename, int index) = 0;

    // 析构函数必须是虚函数
    // 通过基类指针来调用析构函数的时候会用到
    virtual ~Module() = 0;
};

// 用工厂模式产生各个Module
// 工厂类本身采用单例模式，免去构造对象这一冗余的步骤
class ModuleFactory
{
public:
    static ModuleFactory& getInstance();

    // 手动注册Module的接口
    // 注册的时候需要传入全局函数
    // 返回错误代码
    // 函数指针的参数为构建该Module需要的参数，返回值为基类指针
    int registerModule(std::string name, Module* (*pfunc)(std::string));

    // 根据名字删除已经注册的模块
    int deleteModule(std::string name);

    // 构建Module的接口
    // 例如：ModuleFactory::getInstance().create("conv", "3 64 7 2 3")
    //      表示创建卷积核为7x7，步长为2，padding为3，channel从3变到64的卷积层
    // 例如：ModuleFactory::getInstance().create("relu", "")表示构建relu层
    Module* create(std::string name, std::string options);
    // 等价于create(name, "")
    Module* create(std::string name);

private:
    // 注册一些最基本的Module
    // 目前已经实现的有：
    // conv（含有融合BN的接口）
    // relu
    // maxpool
    // upsample
    // bn
    ModuleFactory();

    // 从name到函数指针的映射表
    std::unordered_map<std::string, Module* (*)(std::string)> table;
};

} // namespace snn

#endif
