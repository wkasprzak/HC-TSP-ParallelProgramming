#ifndef DATAGENERATOR_H_
#define DATAGENERATOR_H_

#include <string>

class DataGenerator {

public:
    void generateData(const std::string& baseFileName, unsigned int numberOfCities, int leftConstraint, int rightConstraint, int numInstances);
};

#endif
