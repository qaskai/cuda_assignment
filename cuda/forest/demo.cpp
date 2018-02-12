#include <cuda.h>
#include <cstdio>

void set_roots(int N, CUdeviceptr parents);

static int tree_size = 2048;

void gen_forest(int trees_number, int* parents) {
    for (int k=0; k<trees_number; ++k) {
        // set root parent to itself
        parents[k*tree_size] = k*tree_size;
        // tree is a line
        for (int i=1; i<tree_size; ++i) {
            parents[k*tree_size + i] = k*tree_size + i - 1;
        }
    }
}

void test_correctness(int trees_number, int* parents) {
    for (int i=0; i<trees_number*tree_size; ++i) {
        if (parents[i] != i - (i%tree_size)) {
            printf("FAIL: node %d has parent %d, expected %d\n", i, parents[i], (i-(i%tree_size) ));
        }
    }
}

int main() {

    cuInit(0);

    CUdevice device;
    CUcontext context;

    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) { printf("cuDeviceGet\n"); exit(-1);}  
    if (cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device) != CUDA_SUCCESS) { printf("cuCtxCreate fail\n"); exit(-1); }    

    int trees_number;
    if (scanf("%d", &trees_number) <= 0) {printf("scanf fail\n"); exit(-1); }
    int N = tree_size * trees_number;

    int* parents_table;
    CUdeviceptr dev_parents;
    
    if (cuMemAllocHost((void**)&parents_table, N* sizeof(int)) != CUDA_SUCCESS) {printf("cuMemAllocHost fail parents_table"); exit(-1);}
    if (cuMemAlloc(&dev_parents, N * sizeof(int)) != CUDA_SUCCESS) { printf("cuMemAlloc dev_parents fail\n"); exit(-1);}
    
    gen_forest(trees_number, parents_table);
    if (cuMemcpyHtoD(dev_parents, parents_table, N* sizeof(int)) != CUDA_SUCCESS) { printf("cuMemcpyHtoD dev_parents\n"); exit(-1);}
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync fail\n"); exit(-1);}

    set_roots(N, dev_parents);

    if (cuMemcpyDtoH(parents_table, dev_parents, N* sizeof(int)) != CUDA_SUCCESS) 
        { printf("cuMemcpyDtoH fail\n"); exit(-1); }
    if (cuCtxSynchronize() != CUDA_SUCCESS) { printf("sync fail\n"); exit(-1);}

    test_correctness(trees_number, parents_table);

    cuMemFree(dev_parents);
    cuMemFreeHost(parents_table);
    cuCtxDestroy(context);
    printf("OK :)\n");
}
