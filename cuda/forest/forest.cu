#include <cstdio>

extern "C" {

__global__
void find_roots(int* parents) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    while (parents[x] != parents[parents[x]]) {
        parents[x] = parents[parents[x]];
    }
}

}
