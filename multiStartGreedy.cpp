#include "multiStartGreedy.h"
#include "utility.h"
#include <cfloat>
#include <vector>
#include <omp.h>

std::pair<double, std::vector<int>> multiStartGreedyTSP(std::vector<City>& cities, double batteryCapacity, int ecoWasteFactor, int attempts) {

    double bestCost = DBL_MAX;
    std::vector<int> bestPath;

    #pragma omp parallel
    {
        double threadBestCost = DBL_MAX;
        std::vector<int> threadBestPath;

        #pragma omp for
        for (int attempt = 0; attempt < attempts; ++attempt) {
            int numberOfCities = cities.size();
            int currentCity = attempt % numberOfCities;

            std::vector<bool> visited(numberOfCities, false);
            std::vector<int> road;
            double currentBattery = batteryCapacity;
            double totalDistance = 0.0;

            road.push_back(currentCity);
            visited[currentCity] = true;

            for (int i = 0; i < numberOfCities - 1; ++i) {
                int nextCity = -1;
                double minCost = DBL_MAX;

                for (int j = 0; j < numberOfCities; ++j) {
                    if (!visited[j]) {
                        double dist = calculateDistance(cities[currentCity], cities[j]);
                        double newBattery = currentBattery - dist;
                        double cost = dist + (newBattery < 0 ? ecoWasteFactor : 0.0);
                        if (cost < minCost) {
                            minCost = cost;
                            nextCity = j;
                        }
                    }
                }

                currentBattery -= calculateDistance(cities[currentCity], cities[nextCity]);
                if (currentBattery < 0)
                    currentBattery = batteryCapacity;

                totalDistance += minCost;
                visited[nextCity] = true;
                road.push_back(nextCity);
                currentCity = nextCity;
            }

            road.push_back(road.front());
            double returnDist = calculateDistance(cities[road[road.size() - 2]], cities[road.back()]);
            totalDistance += returnDist;
            currentBattery -= returnDist;
            if (currentBattery < 0) totalDistance += ecoWasteFactor;

            if (totalDistance < threadBestCost) {
                threadBestCost = totalDistance;
                threadBestPath = road;
            }
        }

        #pragma omp critical
        {
            if (threadBestCost < bestCost) {
                bestCost = threadBestCost;
                bestPath = threadBestPath;
            }
        }
    }

    return {bestCost, bestPath};
}
