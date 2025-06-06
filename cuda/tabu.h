#pragma once

#include <vector>
#include <set>
#include <utility>
#include "City.h"

std::pair<double, std::vector<int>> multiStartTabuSearchBlocked_CUDA(
        const std::vector<City>& cities,
        const std::vector<std::vector<double>>& citiesGraph,
        double batteryCapacity,
        int ecoWasteFactor,
        int attempts,
        const std::set<std::pair<int, int>>& forbiddenEdges
);
