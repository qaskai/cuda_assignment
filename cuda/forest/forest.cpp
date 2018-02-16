#include "cuda.h"
#include <cstdio>
#include <algorithm>
#include <stdlib.h>


void set_roots(int N, CUdeviceptr parents)
{
    CUmodule module = (CUmodule) 0;
    if (cuModuleLoad(&module, "forest.ptx") != CUDA_SUCCESS) { printf("cuModuleLoad problem\n"); exit(-1); }

    CUfunction find_roots;
    if (cuModuleGetFunction(&find_roots, module, "find_roots") != CUDA_SUCCESS) { printf("cannot get func find_roots\n"); exit(-1); }

    int chunk = 8;
    void* args[] = {&N, &chunk, &parents};

    //int blocksX = (N+1023)/1024;
    int blocksX = N/(chunk*1024);
    int threadsBlockX = 1024;

    if (cuLaunchKernel(find_roots, blocksX, 1, 1, threadsBlockX, 1, 1, 0, 0, args, 0) != CUDA_SUCCESS) 
        { printf("cuLaunchKernel problem\n"); exit(-1); }
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync kernel problem\n"); exit(-1); }
}
