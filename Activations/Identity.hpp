#pragma once
#ifndef IDENTITY_HPP
#define IDENTITY_HPP

template <typename T>
class Identity {
public:
    T forward(T x) const {
        return x;
    }

    // T backward(T ) const {
    //     return T(1);
    // }
};

#endif // IDENTITY_HPP