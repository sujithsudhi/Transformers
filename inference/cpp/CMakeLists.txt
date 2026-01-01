cmake_minimum_required(VERSION 3.18)

project(tinystories_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

option(USE_EIGEN "Enable Eigen support if available" ON)
option(USE_OPENBLAS "Enable OpenBLAS support if available" ON)

add_library(tinystories_infer
            src/model/encoder.cpp
            src/model/layers.cpp
            src/model/activations.cpp

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
        if (TARGET OpenBLAS::OpenBLAS)
            target_link_libraries(tinystories_infer PUBLIC OpenBLAS::OpenBLAS)
        else()
            target_include_directories(tinystories_infer PUBLIC ${OpenBLAS_INCLUDE_DIRS})
            target_link_libraries(tinystories_infer PUBLIC ${OpenBLAS_LIBRARIES})
        endif()
        target_compile_definitions(tinystories_infer PUBLIC USE_OPENBLAS)
    endif()
endif()

add_executable(tinystories_app
    tinystories.cpp
)

target_link_libraries(tinystories_app
    PRIVATE tinystories_infer
)

FetchContent_Declare(json
                    GIT_REPOSITORY https://github.com/nlohmann/json.git
                    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(cnpy
                     GIT_REPOSITORY https://github.com/rogersce/cnpy.git
                     GIT_TAG master
)
FetchContent_MakeAvailable(cnpy)


target_link_libraries(tinystories_app
                      PRIVATE
                      nlohmann_json::nlohmann_json
                      cnpy)

target_include_directories(tinystories_infer
    PRIVATE
        ${cnpy_SOURCE_DIR}
)

target_include_directories(tinystories_app
    PRIVATE
        ${cnpy_SOURCE_DIR}
)
