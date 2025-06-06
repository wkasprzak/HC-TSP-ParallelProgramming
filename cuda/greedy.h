#pragma once

#include <vector>
#include <tuple>
#include "City.h"

std::pair<double, std::vector<int>> multiStartGreedyTSP(std::vector<City>& cities, double batteryCapacity, int ecoWasteFactor, int attempts);