#include <cuda.h>
#include <cstdio>
#include <climits>
#include <set>

#include <iostream>
#include <chrono>
using namespace std::chrono;

int* bfs(CUdeviceptr graph, int N);

void gen(int seed, int * graph, int N, int dens){
    srand(seed);

    for (int i = 0; i < N*N; ++i) {
        graph[i] = 0;
    }
    for (int i = 0; i < dens; ++i) {
        int x = rand() % N;
        int y = rand() % N;
        graph[x + y * N] = 1;
        graph[y + x * N] = 1;    
    }
    for (int i = 0; i < N; ++i) {
        graph[i + i * N] = 0;
    }
    for (int i =0; i< N-1; ++i) {
        graph[i*N + i+1] = 1;
        graph[(i+1)*N + i] = 1;
    }
}

void test(int N, std::string test_name) {
    int* graphHost;

    CUdevice device;
    CUcontext context;
    CUmodule module;

    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) { printf("cuDeviceGet\n"); exit(-1); }
    if (cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device) != CUDA_SUCCESS) { printf("cuCtxCreate\n"); exit(-1); }

    if (cuMemAllocHost((void**)&graphHost, N*N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAllocHost(graphHost)\n"); exit(-1); }

    gen(12345, graphHost, N, N/16);


    high_resolution_clock::time_point t1 = high_resolution_clock::now();


    CUdeviceptr graph;

    if (cuMemAlloc(&graph, N*N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAlloc(graph)\n"); exit(-1); }
    if (cuMemcpyHtoD(graph, graphHost, N*N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemcpyHtoD\n"); exit(-1); }
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("no sync\n"); exit(-1); }
    int* rad2 = bfs(graph, N);
    /*for (int i=0; i<N; ++i) {
        printf("%d ", rad2[i]);
    }   
    printf("\n");*/

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << test_name << " cuda implementation took " << duration << " us" << std::endl;

    free(rad2);

    cuMemFreeHost(graphHost);
    cuMemFree(graph);
    cuCtxDestroy(context);
}

int main() {
    cuInit(0);
        
    test(1024*8, "N=1024*8 bfs");
    test(1024*16, "N=1024*16 bfs");
    test(1024*30, "N=1024*30 bfs");

    return 0;
}