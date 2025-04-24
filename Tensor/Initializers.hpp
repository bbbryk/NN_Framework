#pragma once
#ifndef INITIALIZERS_HPP
#define INITIALIZERS_HPP
#include <random>
#include "../Tensor/tensor.hpp"

template <typename T>
void randomInit(Tensor<T, 2>& tensor, T min = -1.0, T max = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min, max);

    for (std::size_t i = 0; i < tensor.rows(); ++i) {
        for (std::size_t j = 0; j < tensor.cols(); ++j) {
            tensor(i, j) = dist(gen);
        }
    }
}

template <typename T>
void zeroInit(Tensor<T, 1>& tensor) {
    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = T(0);
    }
}

#endif // INITIALIZERS_HPP
