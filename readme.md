# 版本信息

## hourglass

* 此版本已经跑通了hourglass，可以得到和pytorch版本相同的结果，模型文件为`./hourglass/hg.txt`，参数文件为`./hourglass/hg.array`。
* 4 stack的hourglass耗时189ms，占用内存112MB。内存占用比较稳定（经过无限循环的测试，一直不停地跑forward，结果是内存占用一直在112MB左右浮动）；耗时比较不稳定，体现在：
  * 编译完成后，在终端中执行`export OMP_NUM_THREADS=4`之后再执行./a.out，耗时才能达到189ms（而且有可能浮动），否则耗时会在200～300ms之间浮动。
  * 出现过以下现象：开机一晚上之后，早晨重新编译并执行`./a.out`（已经输入`export OMP_NUM_THREADS=4`），但耗时在260ms左右。关机重启之后此现象消失。

## LeNet-5

* LeNet-5的模型文件在`./LeNet-5/LeNet-5.txt`，参数文件在`./LeNet-5/LeNet-5.array。



# 编译

* OpenBLAS需要用`USE_OPENMP=1`编译，具体过程见OpenBLAS的`Makefile.rule`



# 运行

* 运行前需要设置环境变量

  ```bash
  export OMP_NUM_THREADS=4
  ```

