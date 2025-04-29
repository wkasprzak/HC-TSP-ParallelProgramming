#include "City.h"
#include "multiStartGreedy.h"
#include "tabuSearch.h"
#include "utility.h"

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <sstream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

std::vector<City> convertJsonToCities(const json& dataFromFile, int numberOfCities) {
    std::vector<City> cities;
    for (int i = 0; i < numberOfCities; ++i) {
        const auto& city = dataFromFile[i];
        cities.emplace_back(city["id"], city["x"], city["y"]);
    }
    return cities;
}

int main(int argc, char* argv[]) {
    std::string outputFileName = "ans.csv";
    std::string avgFileName = "avg.csv";
    std::string routeFileName = "routes.csv";

    std::ofstream outputFile(outputFileName, std::ios::app);
    std::ofstream avgFile(avgFileName, std::ios::app);
    std::ofstream routeFile(routeFileName, std::ios::app);
    if (!outputFile.is_open() || !avgFile.is_open() || !routeFile.is_open()) {
        std::cerr << "Cannot open output files.\n";
        return -1;
    }

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <minCities> <maxCities> <repeats> <attempts>\n";
        return 1;
    }

    int minCities = std::stoi(argv[1]);
    int maxCities = std::stoi(argv[2]);
    int repeats = std::stoi(argv[3]);
    int attempts = std::stoi(argv[4]);

    auto addHeadersIfEmpty = [](std::ofstream& file, const std::string& path, const std::string& header) {
        std::ifstream check(path);
        if (check.peek() == std::ifstream::traits_type::eof()) {
            file << header << "\n";
        }
    };

    addHeadersIfEmpty(outputFile, outputFileName, "numberOfCities;method;iteration;time;ecoWaste");
    addHeadersIfEmpty(avgFile, avgFileName, "numberOfCities;method;avgTime;avgEcoWaste");
    addHeadersIfEmpty(routeFile, routeFileName, "numberOfCities;method;iteration;cost;route");

    std::vector<json> allJsonData(repeats);
    for (int i = 0; i < repeats; ++i) {
        std::ifstream file("./data/data" + std::to_string(i + 1) + ".json");
        if (!file.is_open()) {
            std::cerr << "Cannot open JSON file: data" << (i + 1) << ".json\n";
            return -1;
        }
        file >> allJsonData[i];
        file.close();
    }

    std::vector<std::string> methods = {"Greedy", "TabuSearch", "TabuSearchBlocked"};

    for (int m = minCities; m <= maxCities; m+=10) {
        for (const std::string& method : methods) {
            long long totalTime = 0;
            double totalCost = 0.0;

            for (int p = 0; p < repeats; ++p) {
                std::vector<City> cities = convertJsonToCities(allJsonData[p], m);
                std::vector<std::vector<double>> citiesGraph = generateDistanceMatrix(cities);

                double batteryCapacity = 0;
                for (int i = 0; i < m; i++)
                    batteryCapacity += citiesGraph[0][i];
                batteryCapacity = ceil(batteryCapacity * 0.3);
                int ecoWasteFactor = batteryCapacity;

                auto start = std::chrono::steady_clock::now();

                std::pair<double, std::vector<int>> result;
                if (method == "Greedy") {
                    result = multiStartGreedyTSP(cities, batteryCapacity, ecoWasteFactor, attempts);
                } else if (method == "TabuSearch") {
                    result = multiStartTabuSearch(cities, citiesGraph, batteryCapacity, ecoWasteFactor, attempts);
                } else if (method == "TabuSearchBlocked") {
                    auto g = multiStartGreedyTSP(cities, batteryCapacity, ecoWasteFactor, attempts);
                    std::set<std::pair<int, int>> forbiddenEdges;
                    for (size_t i = 0; i < g.second.size() - 1; ++i) {
                        int a = std::min(g.second[i], g.second[i + 1]);
                        int b = std::max(g.second[i], g.second[i + 1]);
                        forbiddenEdges.emplace(a, b);
                    }
                    result = multiStartTabuSearch(cities, citiesGraph, batteryCapacity, ecoWasteFactor, forbiddenEdges, attempts);
                }

                auto end = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                outputFile << m << ";" << method << ";" << p << ";" << duration << ";" << result.first << "\n";
                routeFile << m << ";" << method << ";" << p << ";" << result.first << ";";
                for (int city : result.second)
                    routeFile << city << ",";
                routeFile << "\n";

                totalTime += duration;
                totalCost += result.first;
            }

            avgFile << m << ";" << method << ";" << (totalTime / repeats) << ";" << (totalCost / repeats) << "\n";
        }
    }

    outputFile.close();
    avgFile.close();
    routeFile.close();
    return 0;
}
