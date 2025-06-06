#include <cuda_runtime.h>
#include <float.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include "City.h"

#define MAX_NUM_CITIES 120

// Constant memory for city coordinates, accessible by all threads
__constant__ double constX[MAX_NUM_CITIES];
__constant__ double constY[MAX_NUM_CITIES];

// One greedy attempt per GPU block
__global__ void greedyKernel(int numCities, double battery, double ecoWaste, double* results, int* routes, int* startCities) {
    int attemptId = blockIdx.x;
    // if (attemptId >= gridDim.x) return;

    // Shared memory allocation
    extern __shared__ int shared[];
    int* bestCityPerThread = shared;
    double* bestCostPerThread = (double*)&shared[blockDim.x];

    // Shared memory for per-block state
    __shared__ bool visited[MAX_NUM_CITIES];
    __shared__ int route[MAX_NUM_CITIES+1];
    __shared__ double totalCost;
    __shared__ double currentBattery;
    __shared__ int currentCity;

    // Initialize from the assigned starting city
    int myStartCity = startCities[attemptId];
    if (threadIdx.x == 0) {
        totalCost = 0.0;
        currentBattery = battery;
        currentCity = myStartCity;
        for (int i = 0; i < numCities; ++i) visited[i] = false;
        visited[currentCity] = true;
        route[0] = currentCity;
    }
    __syncthreads();

    for (int step = 1; step < numCities; ++step) {
        __syncthreads();

        // Local copies for thread-safe reads
        int currCityCopy = currentCity;
        double currBatteryCopy = currentBattery;

        // Each thread searches for the best next city
        int bestCityLocal = -1;
        double bestCostLocal = DBL_MAX;

        for (int city = threadIdx.x; city < numCities; city += blockDim.x) {
            if (!visited[city]) {
                double dx = constX[currCityCopy] - constX[city];
                double dy = constY[currCityCopy] - constY[city];
                double dist2 = dx * dx + dy * dy;
                double dist = sqrt(dist2);
                double newBattery = currBatteryCopy - dist;
                double cost = dist2 + ((newBattery < 0.0) ? ecoWaste * ecoWaste : 0.0);

                if (cost < bestCostLocal) {
                    bestCostLocal = cost;
                    bestCityLocal = city;
                }
            }
        }

        bestCityPerThread[threadIdx.x] = bestCityLocal;
        bestCostPerThread[threadIdx.x] = bestCostLocal;
        __syncthreads();

        // Parallel reduction to find globally best city in the block
        int activeThreads = blockDim.x / 2;
        for (int offset = activeThreads / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                if (bestCostPerThread[threadIdx.x + offset] < bestCostPerThread[threadIdx.x]) {
                    bestCostPerThread[threadIdx.x] = bestCostPerThread[threadIdx.x + offset];
                    bestCityPerThread[threadIdx.x] = bestCityPerThread[threadIdx.x + offset];
                }
            }
            __syncthreads();
        }

        // Update global state with the best city found
        if (threadIdx.x == 0) {
            int bestCity = bestCityPerThread[0];
            double dx = constX[currentCity] - constX[bestCity];
            double dy = constY[currentCity] - constY[bestCity];
            double dist = sqrt(dx * dx + dy * dy);
            double penalty = 0.0;
            currentBattery -= dist;
            if (currentBattery < 0.0) {
                penalty = ecoWaste;
                currentBattery = battery;
            }
            totalCost += dist + penalty;
            currentCity = bestCity;
            visited[bestCity] = true;
            route[step] = bestCity;
        }
    }

    __syncthreads();

    // Saving results
    if (threadIdx.x == 0) {
        int startCity = route[0];
        double dx = constX[currentCity] - constX[startCity];
        double dy = constY[currentCity] - constY[startCity];
        double returnDist = sqrt(dx * dx + dy * dy);
        double penalty = 0.0;
        currentBattery -= returnDist;
        if (currentBattery < 0.0) {
            penalty = ecoWaste;
            currentBattery = battery;
        }
        totalCost += returnDist + penalty;
        route[numCities] = startCity;
        results[attemptId] = totalCost;
        for (int i = 0; i <= numCities; ++i) {
            routes[attemptId * (numCities + 1) + i] = route[i];
        }
    }
}

std::pair<double, std::vector<int>> multiStartGreedyTSP(std::vector<City>& cities, double batteryCapacity, int ecoWasteFactor, int attempts) {
    int n = cities.size();
    int routeSize = n + 1;

    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = cities[i].getX();
        y[i] = cities[i].getY();
    }

    // Copy city coordinates to constant memory on GPU
    cudaMemcpyToSymbol(constX, x.data(), sizeof(double) * n);
    cudaMemcpyToSymbol(constY, y.data(), sizeof(double) * n);

    // Generate random starting cities for each attempt
    std::vector<int> h_startCities(attempts);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < attempts; ++i) {
        h_startCities[i] = dist(rng);
    }

    // Allocate and copy start cities to GPU
    int* d_startCities;
    cudaMalloc(&d_startCities, sizeof(int) * attempts);
    cudaMemcpy(d_startCities, h_startCities.data(), sizeof(int) * attempts, cudaMemcpyHostToDevice);

    // Allocate memory for results and routes on GPU
    double* d_results;
    int* d_routes;
    cudaMalloc(&d_results, sizeof(double) * attempts);
    cudaMalloc(&d_routes, sizeof(int) * attempts * routeSize);

    // Maxi no. of threads per block for the current kernel
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, greedyKernel);
    int threadsPerBlock = attr.maxThreadsPerBlock;
    // Required shared memory
    size_t sharedMem = threadsPerBlock * (sizeof(int) + sizeof(double));
    greedyKernel<<<attempts, threadsPerBlock, sharedMem>>>(n, batteryCapacity, ecoWasteFactor, d_results, d_routes, d_startCities);

    // Wait for GPU computation to finish
    cudaDeviceSynchronize();

    // Retrieve results from GPU
    std::vector<double> h_results(attempts);
    cudaMemcpy(h_results.data(), d_results, sizeof(double) * attempts, cudaMemcpyDeviceToHost);

    int bestIdx = 0;
    double bestCost = h_results[0];
    for (int i = 1; i < attempts; ++i) {
        if (h_results[i] < bestCost) {
            bestCost = h_results[i];
            bestIdx = i;
        }
    }

    std::vector<int> bestRoute(routeSize);
    cudaMemcpy(bestRoute.data(), d_routes + bestIdx * routeSize, sizeof(int) * routeSize, cudaMemcpyDeviceToHost);

    cudaFree(d_results);
    cudaFree(d_routes);
    cudaFree(d_startCities);

    return {bestCost, bestRoute};
}
