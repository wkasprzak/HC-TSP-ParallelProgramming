#ifndef CITY_H
#define CITY_H

class City {
private:
    int id;
    double x;
    double y;
public:
    City(int id, double x, double y);
    int getId() const;
    double getX() const;
    double getY() const;
};

#endif
