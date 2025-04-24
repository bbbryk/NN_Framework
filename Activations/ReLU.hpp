#pragma once
#ifndef RELU_HPP
#define RELU_HPP
template <typename T>
class ReLU {
public:
    T forward(T x) const {
        return x > T(0) ? x : T(0);
    }

    // T backward(T y) const {
    //     return y > T(0) ? T(1) : T(0);
    // }
};
#endif // RELU_HPP