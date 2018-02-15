#include <cuda.h>
#include <cstdio>
#include <climits>
#include <set>

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

int main() {
    cuInit(0);
    CUdevice device;
    CUcontext context;
    CUmodule module;

    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) { printf("cuDeviceGet\n"); exit(-1); }
    if (cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device) != CUDA_SUCCESS) { printf("cuCtxCreate\n"); exit(-1); }

    int N = 1024 * 16;
    int SIZE = N * N;
    int * graphHost;

    if (cuMemAllocHost((void**)&graphHost, SIZE * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAllocHost(graphHost)\n"); exit(-1); }

    gen(12345, graphHost, N, 500);

    CUdeviceptr graph;

    if (cuMemAlloc(&graph, SIZE * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAlloc(graph)\n"); exit(-1); }
    if (cuMemcpyHtoD(graph, graphHost, SIZE * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemcpyHtoD\n"); exit(-1); }

    cuCtxSynchronize();

    int* rad2 = bfs(graph, N);
    for (int i=0; i<N; ++i) {
        printf("%d ", rad2[i]);
    }   
    printf("\n");

    free(rad2);

    cuMemFreeHost(graphHost);
    cuMemFree(graph);
    cuCtxDestroy(context);

    return 0;
}