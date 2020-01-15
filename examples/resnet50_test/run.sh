g++ resnet50.cpp \
-I ../../build/install/include \
-L ../../build/install/lib \
-lnn-static -std=c++11 -fopenmp -O3 \
~/open_source_projects/openblas-install/lib/libopenblas.a

export OMP_NUM_THREADS=4
./a.out
rm a.out
