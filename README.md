# SNN

SNN是一个CPU端的小型CNN前向传播框架。除了加载网络参数，它几乎不消耗额外内存，适合在内存受限的系统上运行。SNN的速度和在CPU上运行的pytorch相当。利用SNN自带的OP，可以方便地构建CNN模型，同时它具有良好的可扩展性，可以方便的添加自定义OP。

---

### 优缺点

* ✔️ 除加载网络参数外，几乎无额外内存开销
* ✔️ 以图结构组织OP，支持多输入/多输出
* ✔️ 自动合并连续的卷积和BN层，减少时间开销
* ✔️ 除了矩阵乘法，其余所有代码只使用了C++标准库
* ✔️ 可以方便地添加自定义OP
* ✖️ 依赖OpenBLAS的矩阵乘法
* ✖️ 暂无从pytorch/tensorflow自动转换模型参数的工具
* ✖️ 目前仅可运行于CPU端

---

### 编译及安装

* 安装OpenBLAS库 ( OpenBLAS在`USE_OPENMP=1` 的情况下编译，SNN才能获得最高的性能。具体过程见OpenBLAS的`Makefile.rule`)

* 在`CMakeLists.txt`中设置OpenBLAS头文件的目录 ( 默认为`/opt/OpenBLAS/include` )

  ```txt
  CMakeLists.txt
  ...
  # 设置OpenBLAS的头文件目录
  set(OPENBLAS_INCLUDE "your path")
  ...
  ```

* 假设当前目录为SNN的根目录，执行 ( 需要cmake和make ) :

  ```bash
  mkdir build
  cd build
  cmake ..
  make
  ```

* 默认安装在`build/install`下

  ```bash
  make install
  ```

---

### 性能对比

* 运行前需要设置环境变量 ( 根据具体的环境不同，最优的线程数可能有所变化 )

  ```bash
  export OMP_NUM_THREADS=4
  ```

* 和pytorch的性能对比，以下数据在`Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz`和`pytorch=1.1.0`、`torchvision=0.3.0`的条件下测试得到:

  * **resnet50** ( torchvision官方实现，输入为224 * 224的RGB图像，时间开销为前向传播100次的平均值 )

    |         | 时间开销(ms) | 内存开销(MB) |
    | :-----: | :----------: | :----------: |
    | pytorch |    64.35     |    281.89    |
    |   SNN   |    40.48     |    122.96    |

  * **hourglass** ( 输入为256 * 256的RGB图像，时间为前向传播100次的平均值 )

    |         | 时间开销(ms) | 内存开销(MB) |
    | :-----: | :----------: | :----------: |
    | pytorch |    268.86    |    236.79    |
    |   SNN   |    168.94    |    103.50    |

---

### Examples

`examples/`下有3个使用SNN构建网络的例程，分别是`examples/resnet50`、`examples/hourglass`、`examples/LeNet-5`，编译和运行这些程序需要预先安装opencv。SNN并不依赖于opencv，但SNN本身不包含加载图片的接口，所以构建这些程序需要使用opencv读取图片。每个例子都包含一个`CMakeLists.txt`，以下路径需要被正确设置 :

* opencv的头文件目录，默认为`/usr/local/include/opencv4`
* opencv的库目录，默认为`/usr/local/lib`
* OpenBLAS的库目录，默认为`/opt/OpenBLAS/lib`

#### resnet50

( 模型参数 : https://pan.baidu.com/s/1DBz8I9eKQZz6gRAjCfhOnQ , 提取码 : wp36，放在`examples/resnet50/data`下 )

进入`examples/resnet50`下，执行 :

```bash
mkdir build
cd build
cmake ..
make
./resnet50 ../data/demo.png ../data/resnet50.txt ../data/resnet50.array ../data/labels.txt
```

将会得到以下输出 :

```bash
forward time: xxx us
class id: 285
class name: Egyptian cat
probability: 0.414304
```

#### hourglass

( 模型参数 : https://pan.baidu.com/s/1KIiag7wo2SuWZ6sQLQEFXg , 提取码 : 58sf，放在`examples/hourglass/data`下 )

进入`examples/hourglass`下，执行 :

```bash
mkdir build
cd build
cmake ..
make
./hg ../data/demo.png output.png ../data/hg.txt ../data/hg.array
```

将会得到以下输出 :

```bash
joint name: (location), p: probability
right ankle: (120,220), p: 0.846206
right knee: (120,176), p: 0.820895
right hip: (112,128), p: 0.835805
left hip: (128,128), p: 0.824459
left knee: (132,180), p: 0.833643
left ankle: (140,228), p: 0.851621
pelvis: (120,128), p: 0.855892
thorax: (120,60), p: 0.91914
upper neck: (120,48), p: 0.912966
head top: (116,8), p: 0.875961
right wrist: (92,124), p: 0.877238
right elbow: (104,92), p: 0.828063
right shoulder: (104,60), p: 0.88383
left shoulder: (136,60), p: 0.87217
left elbow: (148,96), p: 0.885659
left wrist: (136,132), p: 0.890509
forward time: xxx us
```

同时，在`build`目录下会生成`output.png`，绘制了所有检测出的关节点，如下图 (图片来自网络，侵删):

![](https://github.com/likunlun0618/snn/blob/master/files/output.png)

#### LeNet-5

( 模型参数 : https://pan.baidu.com/s/1PFSsqlO8rEkQDS2-ExpqGw , 提取码 : wov4，放在`examples/LeNet-5/data`下 )

进入`examples/LeNet-5`下，执行 :

```bash
mkdir build
cd build
cmake ..
make
./lenet-5 ../data/LeNet-5.txt ../data/LeNet-5.array ../data/demo.png
```

将会得到以下输出 :

```bash
output: 
0: 0.00834194
1: -0.000707727
2: -0.0137485
3: -0.00327828
4: -0.00648744
5: 0.998058
6: -0.00491793
7: -0.00222943
8: 0.0193611
9: -0.0122805
The number in ../data/demo.png is 5
```

---

### 添加自定义OP

`examples/addOP`是一个添加自定义OP的例程，添加的OP是对输入的每一个值计算平方，配置`CMakeLists.txt`的方法和前面3个例子相同 ( 不需要设置opencv的路径 ) ，设置好OpenBLAS的路径后，进入`examples/addOP`并执行以下指令 :

```bash
mkdir build
cd build
cmake ..
make
./pow 3 ../pow.txt
```

将会得到以下输出 :

```bash
3^2 = 9
```
