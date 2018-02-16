#include <cstdio>
#include <stdlib.h>
#include <string>

#include <iostream>
#include <chrono>
using namespace std::chrono;


void gen_forest(int trees_number, int tree_size, int* parents) {
    for (int k=0; k<trees_number; ++k) {
        // set root parent to itself
        parents[k*tree_size] = k*tree_size;
        // tree is a line
        for (int i=1; i<tree_size; ++i) {
            parents[k*tree_size + i] = k*tree_size + i - 1;
        }
    }
}

void test_correctness(int trees_number, int tree_size, int* parents) {
    for (int i=0; i<trees_number*tree_size; ++i) {
        if (parents[i] != i - (i%tree_size)) {
            printf("FAIL: node %d has parent %d, expected %d\n", i, parents[i], (i-(i%tree_size) ));
        }
    }
}


void set_roots(int N, int* parents) {
    #pragma omp parallel for
    for (int i=0; i<N; ++i) {
        while (parents[i] != parents[parents[i]]) {
            parents[i] = parents[parents[i]];
        }
    }
}

void test(int trees_number, int tree_size, std::string test_name) {
	int N = trees_number * tree_size;
    int* parents_table = (int*) malloc(N * sizeof(int));    
    gen_forest(trees_number, tree_size, parents_table);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    set_roots(N, parents_table);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    std::cout << test_name << " openmp implementation took " << duration << " us" << std::endl;
    free(parents_table);
    
}

int main(int argc, char const *argv[])
{
    test(1, 2048*2048*2, "'forest openmp small'");
    test(1, 2048*2048*4, "'forest openmp medium'");
    test(1, 2048*2048*8, "'forest openmp large'");

    //printf("OK :)\n");
    return 0;
}