#pragma once
#ifndef NN_HPP
#define NN_HPP
#include <tuple>
#include <utility> 
#include <cstddef>
#include <iostream>

template <typename... Layers>
class NeuralNetwork {
public:
    std::tuple<Layers...> layers;

    NeuralNetwork()
        : layers(std::make_tuple(Layers()...)) {}

    template <typename Input>
    auto forward(const Input& input) {
        return apply_forward(input, std::make_index_sequence<sizeof...(Layers)>());
    }

    void print_weights() const {
        std::cout << "=== Neural Network Weights ===\n";
        apply_print_weights(std::make_index_sequence<sizeof...(Layers)>());
        std::cout << "==============================\n";
    }

    template <typename... Setters>
    void set_weights(Setters&&... setters) {
        static_assert(sizeof...(Setters) == sizeof...(Layers), "Number of setters must match layers.");
        apply_set_weights(std::forward_as_tuple(setters...), std::make_index_sequence<sizeof...(Layers)>());
    }

private:
    template <typename Input, std::size_t... Indices>
    auto apply_forward(const Input& input, std::index_sequence<Indices...>) {
        auto result = input;
        (..., (result = std::get<Indices>(layers).forward(result)));
        return result;
    }

    template <std::size_t... Indices>
    void apply_print_weights(std::index_sequence<Indices...>) const {
        (..., print_layer_weights<Indices>());
    }

    template <std::size_t Index>
    void print_layer_weights() const {
        std::cout << "Layer " << Index << ":\n";
        std::get<Index>(layers).print_weights();
    }

    template <typename Tuple, std::size_t... Indices>
    void apply_set_weights(Tuple&& setters, std::index_sequence<Indices...>) {
        (..., std::get<Indices>(setters)(std::get<Indices>(layers)));
    }
};

#endif // NN_HPP