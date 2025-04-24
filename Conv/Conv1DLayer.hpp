#pragma once
#ifndef CONV1D_HPP
#define CONV1D_HPP
#include "../Tensor/tensor.hpp"
#include "../Tensor/Initializers.hpp"
#include "../Activations/Identity.hpp"
#include <cassert>

template <
    typename T,
    std::size_t InSize,
    std::size_t KernelSize,
    std::size_t OutChannels,
    typename Activation = Identity<T>
>
class Conv1D {
public:
    using InputTensor = Tensor<T, 1>;                         // [InSize]
    using OutputTensor = Tensor<T, 1>;                        // [InSize * OutChannels]

    static constexpr std::size_t Padding = (KernelSize - 1) / 2;
    static constexpr std::size_t OutputSize = InSize;

    Conv1D()
        : filters(OutChannels, KernelSize), biases(OutChannels), activation() {
        static_assert(KernelSize % 2 == 1, "Kernel size must be odd for symmetric padding.");
        randomInit(filters, T(-0.1), T(0.1));
        zeroInit(biases);
    }

    OutputTensor forward(const InputTensor& input) {
        OutputTensor output(OutChannels * OutputSize);

        //TODO: change to eigen lib matrix multiplication
        for (std::size_t ch = 0; ch < OutChannels; ++ch) {
            for (std::size_t i = 0; i < OutputSize; ++i) {
                T sum = biases[ch];
                for (std::size_t k = 0; k < KernelSize; ++k) {
                    std::size_t input_index = i + k;
                    if (input_index < Padding || input_index >= InSize + Padding) {
                        sum += T(0); 
                    } else {
                        sum += filters(ch, k) * input[input_index - Padding];
                    }
                }
                output[ch * OutputSize + i] = activation.forward(sum);
            }
        }

        last_input = input;
        last_output = output;
        return output;
    }

    void print_weights() const {
        std::cout << "Conv1D Weights [Channels: " << OutChannels
                  << ", KernelSize: " << KernelSize << "]\n";
        for (std::size_t ch = 0; ch < OutChannels; ++ch) {
            std::cout << "Filter " << ch << ": [ ";
            for (std::size_t k = 0; k < KernelSize; ++k) {
                std::cout << filters(ch, k);
                if (k + 1 < KernelSize) std::cout << ", ";
            }
            std::cout << " ]\n";
        }
        std::cout << std::endl;
    }

    void set_weights(const T new_weights[OutChannels][KernelSize], const T new_biases[OutChannels]) {
        for (std::size_t ch = 0; ch < OutChannels; ++ch) {
            for (std::size_t k = 0; k < KernelSize; ++k) {
                filters(ch, k) = new_weights[ch][k];
            }
            biases[ch] = new_biases[ch];
        }
    }

private:
    Tensor<T, 2> filters;   // [OutChannels, KernelSize]
    Tensor<T, 1> biases;    // [OutChannels]
    Activation activation;

    InputTensor last_input;
    OutputTensor last_output;
};

#endif // CONV1D_HPP