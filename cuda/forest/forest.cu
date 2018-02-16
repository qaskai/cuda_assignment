#include <cstdio>

extern "C" {

__global__
void find_roots(int N, int chunk, int* parents) {
	int jump = N/chunk;
  	
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    bool flag = true;

    while (flag) {
    	flag = false;
    	for (int i=0; i<chunk; ++i) {
    		if (parents[x + i*jump] != parents[parents[x+ i*jump]]) {
    			parents[x + i*jump] = parents[parents[x+ i*jump]];
    			flag = true;
    		}
    	}
    }
}

}
