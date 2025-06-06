#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class City {
public:
    int id;
    double x;
    double y;

public:
    CUDA_CALLABLE City() : id(0), x(0), y(0) {}
    CUDA_CALLABLE City(int id_, double x_, double y_) : id(id_), x(x_), y(y_) {}

    CUDA_CALLABLE int getId() const { return id; }
    CUDA_CALLABLE double getX() const { return x; }
    CUDA_CALLABLE double getY() const { return y; }
};
