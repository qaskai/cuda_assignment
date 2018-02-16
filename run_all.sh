#!/bin/bash
BASE=`pwd`
rm $BASE/output

set -e
OUT_FILE="${BASE}/output"

cd $BASE/cuda/forest
make
./solution.x >> $OUT_FILE
make clean
cd $BASE/openmp/forest
g++ -std=c++11 -fopenmp forest.cpp -o program
./program >> $OUT_FILE
rm program
echo "" >> $OUT_FILE

cd $BASE/cuda/bfs
make
./solution.x >> $OUT_FILE
make clean
cd $BASE/openmp/bfs
g++ -std=c++11 -fopenmp bfs.cpp -o program
./program >> $OUT_FILE
rm program
echo "" >> $OUT_FILE

TEST_DIR="${BASE}/tests/subtable_tests"
cd $BASE/cuda/subtable
make
./solution.x < $TEST_DIR/test1 >> $OUT_FILE
./solution.x < $TEST_DIR/test2 >> $OUT_FILE
./solution.x < $TEST_DIR/test3 >> $OUT_FILE
make clean
cd $BASE/openmp/subtable
g++ -std=c++11 -fopenmp subtable.cpp -o program
./program < $TEST_DIR/test1 >> $OUT_FILE
./program < $TEST_DIR/test2 >> $OUT_FILE
rm program

printf "\nOpen output file to browse the results.\n\n"