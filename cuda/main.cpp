#include "City.h"
#include "greedy.h"
#include "tabu.h"

#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <cuda_runtime.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

constexpr double BATTERY_FACTOR = 0.3;

// Converts parsed JSON data into a vector of City objects
std::vector<City> convertJsonToCities(const json& dataFromFile, int numberOfCities) {
    std::vector<City> cities;
    for (int i = 0; i < numberOfCities; ++i) {
        const auto& city = dataFromFile[i];
        cities.emplace_back(city["id"], city["x"], city["y"]);
    }
    return cities;
}

// Generates a full distance matrix for all city pairs
std::vector<std::vector<double>> generateDistanceMatrix(const std::vector<City>& cities) {
    int n = cities.size();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = (i == j) ? 0.0 : std::hypot(
                    cities[i].getX() - cities[j].getX(),
                    cities[i].getY() - cities[j].getY()
            );
    return matrix;
}

// Warms up GPU kernels by executing each algorithm once with dummy data
void warmUpKernels() {
    std::vector<City> warmupCities;
    for (int i = 0; i < 10; ++i)
        warmupCities.emplace_back(i, i * 10.0, i * 10.0);

    auto warmupGraph = generateDistanceMatrix(warmupCities);

    double battery = 0.0;
    for (int i = 0; i < 10; ++i)
        battery += warmupGraph[0][i];
    battery = std::ceil(battery * BATTERY_FACTOR);
    int ecoWaste = static_cast<int>(battery);

    multiStartGreedyTSP(warmupCities, battery, ecoWaste, 1);
    cudaDeviceSynchronize();

    std::set<std::pair<int, int>> emptyForbidden;
    multiStartTabuSearchBlocked_CUDA(warmupCities, warmupGraph, battery, ecoWaste, 1, emptyForbidden);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    // Output file paths
    const std::string outputFileName = "ans.csv";
    const std::string avgFileName = "avg.csv";
    const std::string routeFileName = "routes.csv";

    std::ofstream outputFile(outputFileName, std::ios::app);
    std::ofstream avgFile(avgFileName, std::ios::app);
    std::ofstream routeFile(routeFileName, std::ios::app);
    if (!outputFile || !avgFile || !routeFile) {
        std::cerr << "Cannot open output files.\n";
        return -1;
    }

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <minCities> <maxCities> <repeats> <attempts>\n";
        return 1;
    }

    const int minCities = std::stoi(argv[1]);
    const int maxCities = std::stoi(argv[2]);
    const int repeats   = std::stoi(argv[3]);
    const int attempts  = std::stoi(argv[4]);

    // Append CSV headers if files are empty
    auto addHeadersIfEmpty = [](std::ofstream& file, const std::string& path, const std::string& header) {
        std::ifstream check(path);
        if (check.peek() == std::ifstream::traits_type::eof()) {
            file << header << "\n";
        }
    };

    addHeadersIfEmpty(outputFile, outputFileName, "numberOfCities;method;iteration;time;ecoWaste");
    addHeadersIfEmpty(avgFile, avgFileName, "numberOfCities;method;avgTime;avgEcoWaste");
    addHeadersIfEmpty(routeFile, routeFileName, "numberOfCities;method;iteration;cost;route");

    // Load all input JSON data into memory
    std::vector<json> allJsonData(repeats);
    for (int i = 0; i < repeats; ++i) {
        std::ifstream file("./data/data" + std::to_string(i + 1) + ".json");
        if (!file) {
            std::cerr << "Cannot open JSON file: data" << (i + 1) << ".json\n";
            return -1;
        }
        file >> allJsonData[i];
    }

    const std::vector<std::string> methods = {"Greedy", "TabuSearch", "TabuSearchBlocked"};

    // Warm up GPU before benchmarking
    warmUpKernels();

    // Main experiment loop
    for (int m = minCities; m <= maxCities; m += 10) {
        for (const std::string& method : methods) {
            long long totalTime = 0;
            double totalCost = 0.0;
            int validResults = 0;

            for (int p = 0; p < repeats; ++p) {
                std::vector<City> cities = convertJsonToCities(allJsonData[p], m);
                auto citiesGraph = generateDistanceMatrix(cities);

                double batteryCapacity = 0.0;
                for (int i = 0; i < m; ++i)
                    batteryCapacity += citiesGraph[0][i];
                batteryCapacity = std::ceil(batteryCapacity * BATTERY_FACTOR);
                int ecoWasteFactor = static_cast<int>(batteryCapacity);

                auto start = std::chrono::steady_clock::now();

                std::pair<double, std::vector<int>> result;

                if (method == "Greedy") {
                    result = multiStartGreedyTSP(cities, batteryCapacity, ecoWasteFactor, attempts);
                    cudaDeviceSynchronize();
                } else if (method == "TabuSearch") {
                    result = multiStartTabuSearchBlocked_CUDA(cities, citiesGraph, batteryCapacity, ecoWasteFactor, attempts, {});
                    cudaDeviceSynchronize();
                } else if (method == "TabuSearchBlocked") {
                    auto greedyResult = multiStartGreedyTSP(cities, batteryCapacity, ecoWasteFactor, attempts);
                    std::set<std::pair<int, int>> forbiddenEdges;
                    for (size_t i = 0; i < greedyResult.second.size() - 1; ++i) {
                        int a = std::min(greedyResult.second[i], greedyResult.second[i + 1]);
                        int b = std::max(greedyResult.second[i], greedyResult.second[i + 1]);
                        forbiddenEdges.emplace(a, b);
                    }
                    result = multiStartTabuSearchBlocked_CUDA(cities, citiesGraph, batteryCapacity, ecoWasteFactor, attempts, forbiddenEdges);
                    cudaDeviceSynchronize();
                }

                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                if (result.first == std::numeric_limits<double>::max() || std::isinf(result.first))
                    continue;

                validResults++;
                totalTime += duration;
                totalCost += result.first;

                outputFile << m << ";" << method << ";" << p << ";" << duration << ";" << result.first << "\n";
                routeFile << m << ";" << method << ";" << p << ";" << result.first << ";";
                for (int city : result.second)
                    routeFile << city << ",";
                routeFile << "\n";
            }

            if (validResults > 0) {
                double avgTimeUs = static_cast<double>(totalTime) / validResults;
                double avgCost   = totalCost / validResults;
                avgFile << m << ";" << method << ";" << avgTimeUs << ";" << avgCost << "\n";
            }
        }
    }

    // Cleanup
    outputFile.close();
    avgFile.close();
    routeFile.close();
    return 0;
}
