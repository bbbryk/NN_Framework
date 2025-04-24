#pragma once
#pragma once
#ifndef SIGMOID_HPP
#define SIGMOID_HPP
#include <cmath>

template <typename T>
class Sigmoid {
public:
    T forward(T x) const {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    }

    // T backward(T y) const {
    //     return y * (static_cast<T>(1) - y);
    // }
};

#endif // SIGMOID_HPP