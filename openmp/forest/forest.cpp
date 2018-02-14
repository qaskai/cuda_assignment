#include <cstdio>
#include <stdlib.h>

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


void set_roots(int N, int* parents) {
    #pragma omp parallel for
    for (int i=0; i<N; ++i) {
        while (parents[i] != parents[parents[i]]) {
            parents[i] = parents[parents[i]];
        }
    }
}

int main(int argc, char const *argv[])
{
    int trees_number;
    scanf("%d", &trees_number);
    int N = trees_number * tree_size;
    
    int* parents_table = (int*) malloc(N * sizeof(int));    
    gen_forest(trees_number, parents_table);

    set_roots(N, parents_table);

    test_correctness(trees_number, parents_table);

    printf("OK :)\n");
    return 0;
}