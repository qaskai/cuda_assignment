#include <cstdio>
#include <cstdlib>

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

int main(int argc, char const *argv[])
{
    
    int N = 1024*16;

    int* graph = (int*) malloc(N*N * sizeof(int));

    gen(12345, graph, N, 500);

    int* lev = bfs(N, graph);
    for (int i=0; i<N; ++i) {
        printf("%d ", lev[i]);
    }
    printf("\n");

    free(lev);
    free(graph);

    return 0;
}