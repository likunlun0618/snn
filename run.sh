export OMP_NUM_THREADS=4

cd src

g++ hg_test.cpp net.cpp parse.cpp module.cpp tensor.cpp storage.cpp image.cpp \
convolution.cpp relu.cpp maxpool.cpp reshape.cpp linear.cpp upsample.cpp bn.cpp \
addition.cpp \
-I ~/open_source_projects/OpenBLAS-openmp-install/include \
-I /usr/local/include/opencv4 \
~/open_source_projects/OpenBLAS-openmp-install/lib/libopenblas.a \
-L /usr/local/lib/libopencv* \
-fopenmp -lpthread -O3

mv a.out ../

cd ..

# hourglass test (g++ hg_test.cpp ...)
./a.out hourglass/hg.txt hourglass/hg.array hourglass/input.array hourglass/output.array

# hourglass demo (g++ hourglass.cpp ...)
# ./a.out hourglass/demo.png output.png hourglass/hg.txt hourglass/hg.array

# LeNet-5 demo (g++ lenet-5.cpp ...)
# ./a.out LeNet-5/LeNet-5.txt LeNet-5/LeNet-5.array LeNet-5/demo.png
