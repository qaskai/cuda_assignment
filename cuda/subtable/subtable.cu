#include <cstdio>

extern "C" {

__global__
void find_maxes(int N, int* table, int* max_table) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x >= N || y >= N) return;

	int max_sum = table[0];
	for (int i=y; i<N; ++i) {
	for (int j=x; j<N; ++j) {
		int tmp = table[i*N + j];
		
		if (y>0)
			tmp -= table[(y-1)*N + j];
		if (x>0)
			tmp -= table[i*N + x-1];
		if (x>0 && y>0)
			tmp += table[(y-1)*N + x-1];

		max_sum = max(max_sum, tmp);
	}}

	max_table[y*N + x] = max_sum;
}

}
