#include <vector>
#include <set>
#include <limits>
#include <ctime>
#include <cstdio>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <float.h>
#include "City.h"

#define MAX_CITIES 150

__device__ double calcCost(const int* route, const double* graph, int n, double battery, double ecoWaste) {
    double cost = 0.0;
    double currentBattery = battery;

    for (int i = 0; i < n; ++i) {
        int from = route[i];
        int to = route[(i + 1) % n];
        double d = graph[from * n + to];

        currentBattery -= d;
        cost += d;

        if (currentBattery < 0.0) {
            cost += ecoWaste;
            currentBattery = battery;
        }
    }

    return cost;
}

__global__ void tabuKernel(const double* graph, int n, double battery, int ecoWaste, int* routes, double* costs, int maxIter, unsigned long long seed, const int* forbidden, int numForbidden, int* globalTabuList) {

    // Declare shared memory buffer manually and cast it into typed arrays
    extern __shared__ char sharedRaw[];
    int* current     = (int*)sharedRaw;
    int* best        = (int*)&current[n];
    int* threadMoveI = (int*)&best[n];
    int* threadMoveJ = (int*)&threadMoveI[blockDim.x];
    double* threadCosts = (double*)&threadMoveJ[blockDim.x];

    // Each block gets its own slice of the global tabu list
    int* tabu = &globalTabuList[blockIdx.x * n * n];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    curandState state;
    curand_init(seed + bid * blockDim.x + tid, 0, 0, &state);

    // Thread 0 initializes the route and tabu list
    if (tid == 0) {
        for (int i = 0; i < n; ++i) current[i] = i;
        for (int i = n - 1; i > 0; --i) {
            int j = curand(&state) % (i + 1);
            int tmp = current[i];
            current[i] = current[j];
            current[j] = tmp;
        }
        for (int i = 0; i < n; ++i) best[i] = current[i];
        for (int i = 0; i < n * n; ++i) tabu[i] = -1;
    }
    __syncthreads();

    // Compute the cost of the initial solution and define tenure (how long stays int tabu)
    double bestCost = calcCost(current, graph, n, battery, ecoWaste);
    int tenure = 10 + (n / 10);

    for (int iter = 0; iter < maxIter; ++iter) {
        double localBestCost = DBL_MAX;
        int bestI = -1, bestJ = -1;

        // Each thread considers a subset of city pairs (i, j) to swap
        for (int i = tid; i < n - 1; i += blockDim.x) {
            for (int j = i + 1; j < n; ++j) {
                int cityA = current[i], cityB = current[j];
                bool skip = false;
                for (int k = 0; k < numForbidden; ++k) {
                    int a = forbidden[2 * k];
                    int b = forbidden[2 * k + 1];
                    if ((a == cityA && b == cityB) || (a == cityB && b == cityA)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) continue;

                // Try swapping cities i and j in a temporary route
                int temp[MAX_CITIES];
                memcpy(temp, current, sizeof(int) * n);
                int tmp = temp[i];
                temp[i] = temp[j];
                temp[j] = tmp;

                // Calculate the cost of the new route
                double newCost = calcCost(temp, graph, n, battery, ecoWaste);
                // Calculate position in the tabu list (simple hash)
                int pos = (i * 31 + j * 17) % (n * n);
                bool isTabu = (pos < n * n && tabu[pos] >= iter);

                if ((!isTabu || newCost < bestCost) && newCost < localBestCost) {
                    localBestCost = newCost;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        // Shared memory to store the best move among all threads
        __shared__ int globalBestI;
        __shared__ int globalBestJ;
        if (tid == 0) {
            globalBestI = -1;
            globalBestJ = -1;
        }
        __syncthreads();

        // Each thread writes its best move and associated cost
        threadCosts[tid] = localBestCost;
        threadMoveI[tid] = bestI;
        threadMoveJ[tid] = bestJ;
        __syncthreads();

        // Thread 0 performs reduction to find the overall best move
        if (tid == 0) {
            double minThreadCost = DBL_MAX;
            int minI = -1, minJ = -1;
            for (int i = 0; i < blockDim.x; ++i) {
                if (threadMoveI[i] >= 0 && threadCosts[i] < minThreadCost) {
                    minThreadCost = threadCosts[i];
                    minI = threadMoveI[i];
                    minJ = threadMoveJ[i];
                }
            }
            globalBestI = minI;
            globalBestJ = minJ;
        }
        __syncthreads();

        if (tid == 0 && globalBestI >= 0) {
            int i = globalBestI, j = globalBestJ;
            int tmp = current[i]; current[i] = current[j]; current[j] = tmp;
            double updatedCost = calcCost(current, graph, n, battery, ecoWaste);
            int pos = (i * 31 + j * 17) % (n * n);
            if (pos < n * n) tabu[pos] = iter + tenure;

            if (updatedCost < bestCost) {
                bestCost = updatedCost;
                for (int k = 0; k < n; ++k) best[k] = current[k];
            }
        }
        __syncthreads();
    }

    // After all iterations, write the best route and its cost to global memory
    if (tid == 0) {
        for (int i = 0; i < n; ++i)
            routes[bid * n + i] = best[i];
        costs[bid] = bestCost;
    }
}

std::pair<double, std::vector<int>> multiStartTabuSearchBlocked_CUDA(const std::vector<City>& cities, const std::vector<std::vector<double>>& graph, double battery, int ecoWaste, int attempts, const std::set<std::pair<int, int>>& forbiddenEdges) {
    int n = cities.size();

    // Flatten graph into a 1D array for CUDA memory (to not use vector<vector>)
    std::vector<double> flatGraph(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flatGraph[i * n + j] = graph[i][j];

    // Prepare forbidden edges
    std::vector<int> forbiddenFlat;
    for (const auto& edge : forbiddenEdges) {
        forbiddenFlat.push_back(edge.first);
        forbiddenFlat.push_back(edge.second);
    }

    // Allocate device memory
    double* dGraph;
    int* dRoutes;
    double* dCosts;
    int* dForbidden;
    int* dTabu;

    cudaMalloc(&dGraph, sizeof(double) * n * n);
    cudaMalloc(&dRoutes, sizeof(int) * attempts * n);
    cudaMalloc(&dCosts, sizeof(double) * attempts);
    cudaMalloc(&dTabu, sizeof(int) * attempts * n * n);
    cudaMemset(dTabu, -1, sizeof(int) * attempts * n * n);

    // Copy flattened graph to GPU
    cudaMemcpy(dGraph, flatGraph.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice);

    // If forbidden edges exist, copy them to device memory
    if (!forbiddenFlat.empty()) {
        cudaMalloc(&dForbidden, sizeof(int) * forbiddenFlat.size());
        cudaMemcpy(dForbidden, forbiddenFlat.data(), sizeof(int) * forbiddenFlat.size(), cudaMemcpyHostToDevice);
    } else {
        dForbidden = nullptr;
    }

    // Kernel configuration
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, tabuKernel);
    int threadsPerBlock = attr.maxThreadsPerBlock;
    int maxIterations = 100;

    int sharedSize = sizeof(int) * (2 * n + 2 * threadsPerBlock) + sizeof(double) * threadsPerBlock;

    // Launch kernel
    tabuKernel<<<attempts, threadsPerBlock, sharedSize>>>(
            dGraph, n, battery, ecoWaste,
            dRoutes, dCosts, maxIterations,
            static_cast<unsigned long long>(time(NULL)),
            dForbidden, static_cast<int>(forbiddenFlat.size() / 2),
            dTabu
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dGraph);
        cudaFree(dRoutes);
        cudaFree(dCosts);
        cudaFree(dTabu);
        if (dForbidden) cudaFree(dForbidden);
        return {std::numeric_limits<double>::max(), std::vector<int>()};
    }

    // Wait for GPU to finish and check for runtime errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dGraph);
        cudaFree(dRoutes);
        cudaFree(dCosts);
        cudaFree(dTabu);
        if (dForbidden) cudaFree(dForbidden);
        return {std::numeric_limits<double>::max(), std::vector<int>()};
    }

    // Copy results back
    std::vector<int> hostRoutes(attempts * n);
    std::vector<double> hostCosts(attempts);

    cudaMemcpy(hostRoutes.data(), dRoutes, sizeof(int) * attempts * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostCosts.data(), dCosts, sizeof(double) * attempts, cudaMemcpyDeviceToHost);

    // Find best solution
    double bestCost = std::numeric_limits<double>::max();
    int bestIdx = -1;

    for (int i = 0; i < attempts; ++i) {
        if (hostCosts[i] < bestCost && hostCosts[i] > 0) {
            bestCost = hostCosts[i];
            bestIdx = i;
        }
    }

    if (bestIdx == -1) {
        cudaFree(dGraph);
        cudaFree(dRoutes);
        cudaFree(dCosts);
        cudaFree(dTabu);
        if (dForbidden) cudaFree(dForbidden);
        return {bestCost, std::vector<int>()};
    }

    std::vector<int> bestRoute(hostRoutes.begin() + bestIdx * n, hostRoutes.begin() + (bestIdx + 1) * n);

    // Cleanup
    cudaFree(dGraph);
    cudaFree(dRoutes);
    cudaFree(dCosts);
    cudaFree(dTabu);
    if (dForbidden) cudaFree(dForbidden);

    return {bestCost, bestRoute};
}