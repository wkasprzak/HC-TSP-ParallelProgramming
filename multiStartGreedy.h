#pragma once
#include "City.h"
#include <vector>
#include <utility>

std::pair<double, std::vector<int>> multiStartGreedyTSP(std::vector<City>& cities, double batteryCapacity, int ecoWasteFactor, int attempts);