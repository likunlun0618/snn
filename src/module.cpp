#include "module.h"
#include "convolution.h"
#include "relu.h"
#include "maxpool.h"
#include "reshape.h"
#include "linear.h"
#include "upsample.h"
#include "bn.h"
#include "addition.h"

namespace snn
{

Module::~Module() {}

int Module::load(float *data, int size, int index) {}

ModuleFactory& ModuleFactory::getInstance()
{
    static ModuleFactory instance;
    return instance;
}

// 注册常用的Module
// 如果有自定义的Module，除非是非常常见的操作，否则不要在构造函数中注册
// 自定义的Module最好是调用registerModule进行注册，不会有效率上的区别
ModuleFactory::ModuleFactory()
{
    registerModule("conv", createConvolution);
    registerModule("relu", createReLU);
    registerModule("maxpool", createMaxPool);
    registerModule("reshape", createReshape);
    registerModule("linear", createLinear);
    registerModule("upsample", createUpsample);
    registerModule("bn", createBatchNorm);
    registerModule("add", createAddition);
}

// 注意如果有同名的Module，后注册的不会覆盖先注册的，而是返回错误代码
int ModuleFactory::registerModule(std::string name, Module* (*pfunc)(std::string))
{
    if (table.find(name) != table.end())
        return -1; // 已经存在同名的模块，注册失败

    table.insert({name, pfunc});
}

int ModuleFactory::deleteModule(std::string name)
{
    if (table.find(name) == table.end())
        return -1; // 未找到待删除的模块，删除失败

    table.erase(name);
}

Module* ModuleFactory::create(std::string name, std::string options)
{
    // 申请构建的Module没有注册，返回空指针
    // 所以如果用create构建Module，一定要判断返回值是否为空
    if (table.find(name) == table.end())
        return NULL;

    return table[name](options);
}

// 方便构建无参的Module，如relu
Module* ModuleFactory::create(std::string name)
{
    return create(name, "");
}

} // namespace snn
