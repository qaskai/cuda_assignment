#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <chrono>
using namespace std::chrono;

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


int* bfs(int N, int* graph) {
    int* levels = (int*) malloc(N * sizeof(int));
    for (int i=0; i<N; ++i)
        levels[i] = -1;
    levels[0] = 0;
    
    bool flag = true;
    int last_level = 0;
    while (flag) {
        flag = false;

        #pragma omp parallel for
        for (int i=0; i<N*N; ++i) {
            if (graph[i] != 0 && levels[i/N] == last_level && levels[i%N] == -1) {
                levels[i%N] = last_level + 1;
                flag = true;
            }
        }
        
        last_level++;
    }

    return levels;
}

void test(int N, std::string test_name) {

    int* graph = (int*) malloc(N*N * sizeof(int));

    gen(12345, graph, N, N/16);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int* lev = bfs(N, graph);
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << test_name << " openmp implementation took " << duration << " us" << std::endl;

    free(lev);
    free(graph);

}

int main(int argc, char const *argv[])
{
	test(1024*8, "N=1024*8 bfs");
    test(1024*16, "N=1024*16 bfs");
    test(1024*30, "N=1024*30 bfs");

    return 0;
}