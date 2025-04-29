#include "utility.h"
#include "City.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdlib>

using json = nlohmann::json;

double calculateDistance(const City& city1, const City& city2) {
    return std::sqrt(std::pow(city1.getX() - city2.getX(), 2) + std::pow(city1.getY() - city2.getY(), 2));
}

double checkEcoWaste(const std::vector<std::vector<double>>& citiesGraph, const std::vector<int>& road, double batteryCapacity, int ecoWasteFactor) {
    double totalDistance = 0.0;
    double batteryLevel = batteryCapacity;
    int numberOfCities = road.size();

    for (int i = 0; i < numberOfCities - 1; ++i) {
        double dist = citiesGraph[road[i]][road[i + 1]];
        batteryLevel -= dist;
        totalDistance += dist;
        if (batteryLevel < 0) {
            totalDistance += ecoWasteFactor;
            batteryLevel = batteryCapacity;
        }
    }

    double dist = citiesGraph[road.back()][road.front()];
    batteryLevel -= dist;
    totalDistance += dist;
    if (batteryLevel < 0)
        totalDistance += ecoWasteFactor;

    return totalDistance;
}


std::vector<City> loadCitiesFromFile(const std::string& filename, int numberOfCities) {
    std::ifstream dataFile(filename);
    if (!dataFile.is_open()) {
        std::cerr << "Unable to open JSON file: " << filename << "\n";
        return {};
    }
    json dataFromFile;
    dataFile >> dataFromFile;
    dataFile.close();

    std::vector<City> cities;
    for (int i = 0; i < numberOfCities; ++i) {
        const auto& oneCity = dataFromFile[i];
        cities.emplace_back(oneCity["id"], oneCity["x"], oneCity["y"]);
    }
    return cities;
}

std::vector<std::vector<double>> generateDistanceMatrix(const std::vector<City>& cities) {
    int n = cities.size();
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = (i == j) ? 0.0 : calculateDistance(cities[i], cities[j]);
    return matrix;
}

void saveRouteToFile(const std::string& filename, int citiesCount, const std::string& method, int iteration,
                     double cost, const std::vector<int>& route) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Cannot open route output file: " << filename << "\n";
        return;
    }
    file << citiesCount << ";" << method << ";" << iteration << ";" << cost << ";";
    for (int city : route)
        file << city << ",";
    file << "\n";
    file.close();
}
