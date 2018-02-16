#include <cuda.h>
#include <cstdio>
#include <climits>
#include <algorithm>

#include <iostream>
#include <chrono>
using namespace std::chrono;


void local_maxes(int N, CUdeviceptr table, CUdeviceptr local_max_table);


void parse_input(int N, int* table) {
    for (int i=0; i<N*N; ++i) {
        if (scanf("%d", &table[i]) <= 0) { printf("scanf error\n"); exit(-1); }
    }
}

int find_max(int N, int* table) {
    int max = INT_MIN;
    for (int i=0; i<N*N; ++i) {
        max = std::max(max, table[i]);
    }
    return max;
}


void partial_sums(int N, int* table) {
    for (int i=1; i<N; ++i) {
        table[i] += table[i-1];
        table[i*N] += table[(i-1) * N];
    }

    for (int i=1; i<N; ++i) {
    for (int j=1; j<N; ++j) {
        int tmp = table[(i-1)*N +j] + table[i*N + j-1];
        tmp -= table[(i-1)*N + j-1];
        table[i*N + j] += tmp;
    }}
}



int main() {
    
    cuInit(0);
    
    CUdevice device;
    CUcontext context;

    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) { printf("cuDeviceGet fail\n"); exit(-1); }
    if (cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device) != CUDA_SUCCESS) { printf("cuCtxCreate fail\n"); exit(-1); }

    int N;
    if (scanf("%d", &N) <= 0) { printf("scanf failed\n"); exit(-1); }

    int* host_table;
    CUdeviceptr table;
    CUdeviceptr local_max_table;

    if (cuMemAllocHost((void**) &host_table, N*N* sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAllocHost fail\n"); exit(-1); }
    if (cuMemAlloc(&table, N*N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAlloc table fail\n"); exit(-1); }
    if (cuMemAlloc(&local_max_table, N*N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAlloc local_max_table fail\n"); exit(-1); }

    parse_input(N, host_table);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    partial_sums(N, host_table);

    if (cuMemcpyHtoD(table, host_table, N*N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemcpyHtoD host_table --> table fail\n"); exit(-1);}
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync fail\n"); exit(-1);}

    local_maxes(N, table, local_max_table);

    if (cuMemcpyDtoH(host_table, local_max_table, N*N * sizeof(int)) != CUDA_SUCCESS) 
        { printf("cuMemcpyDtoH local_max_table --> host_table fail\n"); exit(-1);}
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync fail\n"); exit(-1);}

    int max = find_max(N, host_table);
    //printf("%d\n", max);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "subtable test N=" << N << " cuda implementation took " << duration << " us" << std::endl;

    cuMemFree(table);
    cuMemFree(local_max_table);
    cuMemFreeHost(host_table);
    cuCtxDestroy(context);

    return 0;
}
