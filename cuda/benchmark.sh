#!/bin/bash

MIN_CITIES=10
MAX_CITIES=120
REPEATS=10
ATTEMPTS=10
EXECUTABLE=./tsp_cuda
OUTPUT_CSV="benchmark_results.csv"

echo "Kompilacja"
make clean && make || { echo "Error"; exit 1; }

rm -f ans.csv avg.csv routes.csv $OUTPUT_CSV
echo "mode,real,user,sys" > $OUTPUT_CSV

echo "Running CUDA version..."
/usr/bin/time -f "cuda,%e,%U,%S" $EXECUTABLE $MIN_CITIES $MAX_CITIES $REPEATS $ATTEMPTS 2>> $OUTPUT_CSV

echo -e "\nBenchmark results:"
column -s, -t $OUTPUT_CSV
