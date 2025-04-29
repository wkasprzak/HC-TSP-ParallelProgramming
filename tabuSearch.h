#pragma once
#include "City.h"
#include <vector>
#include <set>
#include <utility>


std::vector<std::pair<double, std::vector<int>>> tabuSearch(const std::vector<City>& cities, const std::vector<std::vector<double>>& citiesGraph, double batteryCapacity, int ecoWasteFactor, const std::set<std::pair<int, int>>& forbiddenEdges = {});

std::pair<double, std::vector<int>> multiStartTabuSearch(const std::vector<City>& cities, const std::vector<std::vector<double>>& citiesGraph, double batteryCapacity, int ecoWasteFactor, const std::set<std::pair<int, int>>& forbiddenEdges, int attempts);

std::pair<double, std::vector<int>> multiStartTabuSearch(const std::vector<City>& cities, const std::vector<std::vector<double>>& citiesGraph, double batteryCapacity, int ecoWasteFactor, int attempts);