#include "DataGenerator.h"
#include "nlohmann/json.hpp"

#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <random>
#include <chrono>

void DataGenerator::generateData(const std::string& baseFileName, unsigned int numberOfCities, int leftConstraint, int rightConstraint, int numInstances) {

    if (leftConstraint > rightConstraint)
        std::swap(leftConstraint, rightConstraint);

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int> dist(leftConstraint, rightConstraint);

    for (int k = 0; k < numInstances; ++k) {
        nlohmann::json dataArray;

        for (int i = 0; i < numberOfCities; i++) {
            nlohmann::json generatedData;
            generatedData["id"] = i;
            generatedData["x"] = dist(gen);
            generatedData["y"] = dist(gen);
            dataArray.push_back(generatedData);
        }

        std::string filePath = "./data/" + baseFileName + std::to_string(k + 1) + ".json";
        std::ofstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Unable to open JSON file: " << filePath << std::endl;
            continue;
        }
        file << dataArray.dump(3);
        file.close();
    }
}