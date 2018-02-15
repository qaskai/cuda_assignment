#include <cstdio>

extern "C" {
__global__
void init(int* levels) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (x!= 0){
        levels[x] = -1;
    }
    else {
        levels[0] = 0;
    }
}

__global__
void bfs(int* levels, int* graph, int level, int* changed, int n) {  
    int y = (blockIdx.x * blockDim.x) + threadIdx.x;
    int x = (blockIdx.y * blockDim.y) + threadIdx.y; 

    bool flag = false;
    if (levels[y] != level) {
        return;
    }
    if (graph[y*n + x] > 0 && levels[x] == -1) {
        flag = true;
        levels[x] = level+1;
    }
    if (flag)
        *changed = 1;
}
}