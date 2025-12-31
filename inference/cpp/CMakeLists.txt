cmake_minimum_required(VERSION 3.18)

project(tinystories_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_EIGEN "Enable Eigen support if available" ON)
option(USE_OPENBLAS "Enable OpenBLAS support if available" ON)

add_library(tinystories_infer
    encoder.cpp
    layers.cpp
    load_params.cpp
)

target_include_directories(tinystories_infer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(tinystories_infer PUBLIC cxx_std_17)

if (USE_EIGEN)
    find_package(Eigen3 QUIET)
    if (Eigen3_FOUND)
        target_link_libraries(tinystories_infer PUBLIC Eigen3::Eigen)
        target_compile_definitions(tinystories_infer PUBLIC USE_EIGEN)
    endif()
endif()

if (USE_OPENBLAS)
    find_package(OpenBLAS QUIET)
    if (OpenBLAS_FOUND)
        target_link_libraries(tinystories_infer PUBLIC OpenBLAS::OpenBLAS)
        target_compile_definitions(tinystories_infer PUBLIC USE_OPENBLAS)
    endif()
endif()

add_executable(tinystories_app
    tinystories.cpp
)

target_link_libraries(tinystories_app
    PRIVATE tinystories_infer
)
