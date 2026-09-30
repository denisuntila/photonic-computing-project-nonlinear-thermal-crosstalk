cmake_minimum_required(VERSION 3.12.0)
project(Simulation LANGUAGES CXX C)

include(./common/cmake-common.cmake)

add_executable(simulation src/main.cpp src/Heat.cpp)
deal_ii_setup_target(simulation)

add_executable(non_linear src/main_non_linear.cpp src/NonLinear.cpp)
deal_ii_setup_target(non_linear)




