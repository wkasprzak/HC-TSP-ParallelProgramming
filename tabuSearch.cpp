#include "tabuSearch.h"
#include "utility.h"
#include <unordered_set>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <float.h>
#include <random>
#include <chrono>
#include <sstream>
#include <omp.h>

// haszowanie do zmniejszenia zużycia zasobów polluksa
uint64_t hashRoute(const std::vector<int>& route) {
    uint64_t hash = 14695981039346656037ull;
    for (int city : route) {
        hash ^= city;
        hash *= 1099511628211ull;
    }
    return hash;
}

std::vector<int> generateRandomTour(int numberOfCities) {
    std::vector<int> tour(numberOfCities);
    for (int i = 0; i < numberOfCities; ++i)
        tour[i] = i;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(tour.begin(), tour.end(), std::default_random_engine(seed));
    return tour;
}

std::pair<double, std::vector<int>> multiStartTabuSearch(const std::vector<City>& cities, const std::vector<std::vector<double>>& citiesGraph, double batteryCapacity, int ecoWasteFactor, const std::set<std::pair<int, int>>& forbiddenEdges, int attempts) {
    double bestCost = DBL_MAX;
    std::vector<int> bestRoute;

    #pragma omp parallel
    {
        double localBestCost = DBL_MAX;
        std::vector<int> localBestRoute;

        #pragma omp for
        for (int i = 0; i < attempts; ++i) {
            auto results = tabuSearch(cities, citiesGraph, batteryCapacity, ecoWasteFactor, forbiddenEdges);
            if (!results.empty()) {
                double cost = results[0].first;
                const auto& route = results[0].second;

                if (cost < localBestCost) {
                    localBestCost = cost;
                    localBestRoute = route;
                }
            }
        }

        #pragma omp critical
        {
            if (localBestCost < bestCost) {
                bestCost = localBestCost;
                bestRoute = localBestRoute;
            }
        }
    }

    return {bestCost, bestRoute};
}

std::vector<std::pair<double, std::vector<int>>> tabuSearch(const std::vector<City>& cities, const std::vector<std::vector<double>>& citiesGraph, double batteryCapacity, int ecoWasteFactor, const std::set<std::pair<int, int>>& forbiddenEdges) {
    int maxIterations = 100;
    size_t numberOfCities = cities.size();
    size_t tabuListSize = numberOfCities;

    std::vector<std::tuple<double, std::vector<int>>> topResults;
    std::unordered_set<uint64_t> tabuSet;
    std::vector<uint64_t> tabuQueue;

    std::vector<int> bestTour = generateRandomTour(numberOfCities);
    std::vector<int> currentTour = bestTour;
    tabuSet.insert(hashRoute(currentTour));
    tabuQueue.push_back(hashRoute(currentTour));

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        std::vector<int> bestNeighbor;
        double minCost = DBL_MAX;

        for (size_t i = 0; i < currentTour.size() - 1; ++i) {
            for (size_t j = i + 1; j < currentTour.size(); ++j) {
                if (forbiddenEdges.count({std::min(currentTour[i], currentTour[j]), std::max(currentTour[i], currentTour[j])}))
                    continue;

                std::vector<int> neighbor = currentTour;
                std::swap(neighbor[i], neighbor[j]);

                uint64_t hash = hashRoute(neighbor);
                if (!tabuSet.count(hash)) {
                    double cost = checkEcoWaste(citiesGraph, neighbor, batteryCapacity, ecoWasteFactor);
                    if (cost < minCost) {
                        minCost = cost;
                        bestNeighbor = neighbor;
                    }
                }
            }
        }

        if (!bestNeighbor.empty()) {
            currentTour = bestNeighbor;
            uint64_t serialized = hashRoute(currentTour);
            tabuSet.insert(serialized);
            tabuQueue.push_back(serialized);
            if (tabuQueue.size() > tabuListSize) {
                tabuSet.erase(tabuQueue.front());
                tabuQueue.erase(tabuQueue.begin());
            }

            bool isDuplicate = false;
            for (const auto& result : topResults) {
                const auto& path = std::get<1>(result);
                if (path == currentTour) {
                    isDuplicate = true;
                    break;
                }
            }

            if (!isDuplicate) {
                double cost = checkEcoWaste(citiesGraph, currentTour, batteryCapacity, ecoWasteFactor);
                topResults.emplace_back(cost, currentTour);
                std::sort(topResults.begin(), topResults.end(), [](const auto& a, const auto& b) {
                    return std::get<0>(a) < std::get<0>(b);
                });
                if (topResults.size() > 3) topResults.pop_back();
            }
        }
    }

    std::clog << "\n--- TOP 3 Routes ---" << std::endl;
    std::vector<std::pair<double, std::vector<int>>> finalResults;
    for (size_t i = 0; i < topResults.size(); ++i) {
        auto route = std::get<1>(topResults[i]);
        double cost = std::get<0>(topResults[i]);
        route.push_back(route.front());
        std::clog << "#" << i + 1 << " Cost: " << cost << "\nOrder of cities: ";
        for (int city : route) std::clog << city << " ";
        std::clog << std::endl;
        finalResults.emplace_back(cost, route);
    }

    return finalResults;
}

std::pair<double, std::vector<int>> multiStartTabuSearch(const std::vector<City>& cities, const std::vector<std::vector<double>>& citiesGraph, double batteryCapacity, int ecoWasteFactor, int attempts) {
    std::set<std::pair<int, int>> emptySet;
    return multiStartTabuSearch(cities, citiesGraph, batteryCapacity, ecoWasteFactor, emptySet, attempts);
}
