#pragma once
#include "City.h"
#include <vector>
#include <string>

double calculateDistance(const City &city1, const City &city2);

double checkEcoWaste(const std::vector<std::vector<double>>& citiesGraph, const std::vector<int>& road, double batteryCapacity, int ecoWasteFactor);

std::vector<City> loadCitiesFromFile(const std::string& filename, int numberOfCities);

std::vector<std::vector<double>> generateDistanceMatrix(const std::vector<City>& cities);

void saveRouteToFile(const std::string& filename, int citiesCount, const std::string& method, int iteration, double cost, const std::vector<int>& route);
