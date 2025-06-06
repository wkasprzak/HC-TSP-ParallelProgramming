CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17 -I. -Iexternal/json
NVCCFLAGS = -O3 -Xcompiler -fopenmp -std=c++17
LDFLAGS = -fopenmp -L/usr/local/cuda/lib64 -lcudart

SRC = main.cpp
CU_SRC = greedy.cu tabu.cu
OBJ = $(SRC:.cpp=.o)
CU_OBJ = $(CU_SRC:.cu=.o)

all: tsp_cuda

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

tsp_cuda: $(OBJ) $(CU_OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o tsp_cuda benchmark_results.csv

