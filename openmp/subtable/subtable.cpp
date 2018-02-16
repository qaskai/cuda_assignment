#include <cstdio>
#include <algorithm>
#include <climits>
#include <cstdlib>

#include <iostream>
#include <chrono>
using namespace std::chrono;

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

void find_local_maxes(int N, int* table, int* local_maxes) {

    #pragma omp parallel for
    for (int a=0; a<N*N; ++a) {
        int y = a/N, x = a%N;
        int max = table[0];
        for (int i=y; i<N; ++i) {
        for (int j=x; j<N; ++j) {
            int tmp = table[i*N + j];
        
            if (y>0)
                tmp -= table[(y-1)*N + j];
            if (x>0)
                tmp -= table[i*N + x-1];
            if (x>0 && y>0)
                tmp += table[(y-1)*N + x-1];

            max = std::max(tmp, max);
        }
        }
        local_maxes[a] = max;
    }
}


int main(int argc, char const *argv[])
{
    int N;
    scanf("%d", &N);
    
    int* table = (int*) malloc(N*N * sizeof(int));
    int* local_maxes = (int*) malloc(N*N * sizeof(int));
    parse_input(N, table);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    partial_sums(N, table);
    find_local_maxes(N, table, local_maxes);

    int max = find_max(N, local_maxes);
    //printf("%d\n", max);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << "test N=" << N << " openmp implementation took " << duration << " us" << std::endl;

    free(table);
    free(local_maxes);

    return 0;
}