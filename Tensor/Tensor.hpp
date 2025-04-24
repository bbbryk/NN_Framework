#pragma once
#ifndef TENSOR_HPP
#define TENSOR_HPP
#include <Eigen/Dense>
#include <cstddef>

template <typename T, int Dim>
class Tensor;

template <typename T>
class Tensor<T, 1> {
public:
    using DataType = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    Tensor() = default;

    explicit Tensor(std::size_t size) : data(size) {}

    std::size_t size() const {
        return data.size();
    }

    T& operator[](std::size_t i) {
        return data(i);
    }

    const T& operator[](std::size_t i) const {
        return data(i);
    }

    const DataType& raw() const { return data; }
    DataType& raw() { return data; }

private:
    DataType data;
};

template <typename T>
class Tensor<T, 2> {
public:
    using DataType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    Tensor() = default;

    Tensor(std::size_t rows, std::size_t cols) : data(rows, cols) {}

    std::size_t rows() const {
        return data.rows();
    }

    std::size_t cols() const {
        return data.cols();
    }

    T& operator()(std::size_t i, std::size_t j) {
        return data(i, j);
    }

    const T& operator()(std::size_t i, std::size_t j) const {
        return data(i, j);
    }

    const DataType& raw() const { return data; }
    DataType& raw() { return data; }

private:
    DataType data;
};
#endif // TENSOR_HPP