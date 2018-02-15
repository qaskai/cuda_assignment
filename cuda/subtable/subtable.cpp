#include "cuda.h"
#include <cstdio>
#include <algorithm>
#include <stdlib.h>


void local_maxes(int N, CUdeviceptr table, CUdeviceptr local_maxes)
{
    CUmodule module;
    if (cuModuleLoad(&module, "subtable.ptx") != CUDA_SUCCESS) { printf("cuModuleLoad fail\n"); exit(-1); }

    CUfunction maxes;
    if (cuModuleGetFunction(&maxes, module, "find_maxes") != CUDA_SUCCESS) { printf("cuGetFunction fail\n"); exit(-1); }

    void* args[] = {&N, &table, &local_maxes};

    int blocks, threads;
    blocks = (N+31) / 32;
    threads = 32;

    if (cuLaunchKernel(maxes, blocks, blocks, 1, threads, threads, 1, 0, 0, args, 0) != CUDA_SUCCESS) {
        printf("cuLaunchKernel fail\n"); exit(-1);
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync fail\n"); exit(-1); }
}
