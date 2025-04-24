cmake_minimum_required(VERSION 3.10)
project(NN_Framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Activations
    ${CMAKE_SOURCE_DIR}/Dense
    ${CMAKE_SOURCE_DIR}/Tensor
    ${CMAKE_SOURCE_DIR}/Conv
)

add_executable(nn_framework main.cpp)
