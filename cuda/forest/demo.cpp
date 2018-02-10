#include <cuda.h>
#include <cstdio>

void set_roots(int N, CUdeviceptr parents);

void gen_random_forest(int N, int* parents) {
    /*to implement*/

}

int main() {
    
    cuInit(0);

    CUdevice device;
    CUcontext context;

    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) { printf("cuDeviceGet\n"); exit(-1);}  
    if (cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device) != CUDA_SUCCESS) { printf("cuCtxCreate fail\n"); exit(-1); }    

    int N;
    if (scanf("%d", &N) <= 0) {printf("scanf fail\n"); exit(-1); }

    int* parents_table;
    CUdeviceptr dev_parents;
    
    if (cuMemAllocHost((void**)&parents_table, N* sizeof(int)) != CUDA_SUCCESS) {printf("cuMemAllocHost fail parents_table"); exit(-1);}
    if (cuMemAlloc(&dev_parents, N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAlloc dev_parents fail\n"); exit(-1);}
    
    gen_random_forest(N, parents_table);

    if (cuMemcpyHtoD(dev_parents, parents_table, N* sizeof(int)) != CUDA_SUCCESS) { printf("cuMemcpyHtoD dev_parents\n"); exit(-1);}
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync fail\n"); exit(-1);}

    set_roots(N, dev_parents);

    if (cuMemcpyDtoH(parents_table, dev_parents, N* sizeof(int)) != CUDA_SUCCESS) 
        { printf("cuMemcpyDtoH fail\n"); exit(-1); }
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync fail\n"); exit(-1);}

    cuMemFree(dev_parents);
    cuMemFreeHost(parents_table);
    cuCtxDestroy(context);
    free(parents_table);

}
