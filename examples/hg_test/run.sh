g++ hg_test.cpp \
-I ../../build/install/include \
-L ../../build/install/lib \
-lnn-static -fopenmp -O3 \
~/open_source_projects/OpenBLAS-openmp-install/lib/libopenblas.a

export OMP_NUM_THREADS=4
./a.out ../hourglass/data/hg.txt ../hourglass/data/hg.array input.array output.array
rm a.out
