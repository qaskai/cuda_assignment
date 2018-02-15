#include "cuda.h"
#include <cstdio>
#include <climits>
#include <algorithm>


int* bfs(CUdeviceptr graph, int n)
{
    CUresult res;
    //cuInit(0);

    CUmodule cuModule = (CUmodule) 0;
    res = cuModuleLoad(&cuModule, "bfs.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);
        exit(1);
    }

    CUfunction init;
    res = cuModuleGetFunction(&init, cuModule, "init");
    if (res != CUDA_SUCCESS) {
        printf("cannot get function1\n");
        exit(1);
    }
    CUfunction bfs;
    res = cuModuleGetFunction(&bfs, cuModule, "bfs");
    if (res != CUDA_SUCCESS) {
        printf("cannot get function2\n");
        exit(1);
    }

    CUdeviceptr levels;
        res = cuMemAlloc(&levels, n*sizeof(int));
        if (res != CUDA_SUCCESS) {
            printf("kernel problem1\n");
            exit(1);
        }
        void* args[] = {&levels};
        res = cuLaunchKernel(init, (n+1023)/1024, 1, 1, 1024, 1, 1, 0, 0, args, 0);
        if (res != CUDA_SUCCESS) {
            printf("kernel problem14321\n");
            exit(1);
        }
        res = cuCtxSynchronize();
        if (res != CUDA_SUCCESS) {
            printf("no sync\n");
            exit(1);
        }


        int* changed;
        res = cuMemAllocHost((void**) &changed, sizeof(int));
        if (res != CUDA_SUCCESS) {
            printf("memory problem1\n");
            exit(1);
        }
        *changed = 1;

int level = 0;
    while (*changed) {
        *changed =0;
        int blocksX = (n+31)/32, blocksY = (n+31)/32;
        int threads = 32;
        void* args[] = {&levels, &graph, &level, &changed, &n};
        res = cuLaunchKernel(bfs, blocksX, blocksY, 1, threads, threads, 1, 0, 0, args, 0);
        if (res != CUDA_SUCCESS) {
            printf("kernel problem1\n");
            exit(1);
        }
        res = cuCtxSynchronize();
        if (res != CUDA_SUCCESS) {
            printf("no sync\n");
            exit(1);
        }
        level++;
    }

    int* result=  (int*) malloc(n* sizeof(int));
    res = cuMemcpyDtoH(result, levels, n*sizeof(int));
    if (res != CUDA_SUCCESS) {
            printf("copy problem D to H\n");
            exit(1);
    }
    return result;
}