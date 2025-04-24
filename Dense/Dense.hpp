#pragma once
#ifndef DENSE_HPP
#define DENSE_HPP

#include "../Tensor/Tensor.hpp"
#include "../Tensor/Initializers.hpp"
#include "../Activations/Identity.hpp"

template <
    typename T,
    std::size_t In,
    std::size_t Out,
    typename Activation = Identity<T> 
>class Dense {
public:
    using InputTensor = Tensor<T, 1>;
    using OutputTensor = Tensor<T, 1>;

    Dense()
        : weights(Out, In), biases(Out),
          grad_weights(Out, In), grad_biases(Out),
          activation() {
        randomInit(weights, T(-0.1), T(0.1));  
        zeroInit(biases);
    }

    OutputTensor forward(const InputTensor& input) {
        OutputTensor output(Out);
        for (std::size_t i = 0; i < Out; ++i) {
            T sum = biases[i];
            for (std::size_t j = 0; j < In; ++j) {
                sum += weights(i, j) * input[j];
            }
            output[i] = activation.forward(sum);
        }
        last_input = input;
        last_output = output;
        return output;
    }




    void print_weights() const {
        std::cout << "Weights [" << Out << " x " << In << "]:" << std::endl;
        for (std::size_t i = 0; i < Out; ++i) {
            std::cout << "[ ";
            for (std::size_t j = 0; j < In; ++j) {
                std::cout << weights(i, j);
                if (j + 1 < In) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        }
        std::cout << std::endl;
    }
    void set_weights(const T new_weights[Out][In], const T new_biases[Out]) {
        for (std::size_t i = 0; i < Out; ++i) {
            for (std::size_t j = 0; j < In; ++j) {
                weights(i, j) = new_weights[i][j];
            }
            biases[i] = new_biases[i];
        }
    }

private:
    Tensor<T, 2> weights;
    Tensor<T, 1> biases;
    Tensor<T, 2> grad_weights;
    Tensor<T, 1> grad_biases;

    Activation activation;
    InputTensor last_input;
    OutputTensor last_output;
};

#endif // DENSE_HPP