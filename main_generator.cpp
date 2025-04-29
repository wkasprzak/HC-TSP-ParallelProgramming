#include "DataGenerator.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <baseFileName> <numCities> <leftBound> <rightBound> <numInstances>\n";
        return 1;
    }

    std::string fileName = argv[1];
    int numCities = std::stoi(argv[2]);
    int left = std::stoi(argv[3]);
    int right = std::stoi(argv[4]);
    int instances = std::stoi(argv[5]);

    DataGenerator generator;
    generator.generateData(fileName, numCities, left, right, instances);

    std::cout << "Generated " << instances << " dataset(s).\n";
    return 0;
}