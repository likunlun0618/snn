export OMP_NUM_THREADS=4
g++ -fopenmp conv_gemm.cpp random.cpp io.cpp -I ~/open_source_projects/OpenBLAS-openmp-install/include \
~/open_source_projects/OpenBLAS-openmp-install/lib/libopenblas.a -lpthread -O3

./a.out

python3 test.py
