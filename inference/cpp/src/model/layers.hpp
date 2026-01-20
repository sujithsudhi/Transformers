#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "utils.hpp"
#include "model.hpp"
#include <cmath>


namespace layers
{

    // Here we define the Linear layer (Dense layer without activation)
class Linear
{
public:
    Linear(const int32_t _inputDim,
            const int32_t _outputDim,
            const float_t* wPtr,
            const float_t* bPtr,
            const std::string _name = "");


    void forward(const RowVec& input, RowVec& output);

    private:
        const int32_t inputDim;
        const int32_t outputDim;
        const std::string name;
        Eigen::Map<const MatrixRM> weights;
        Eigen::Map<const Vector> bias;

};


class LayerNorm
{
public:
    LayerNorm(const float* g_ptr,
              const float* b_ptr,
              int d,
              float eps_);

    void forward(const Vector& input,  Vector& ouput);

private:
    int32_t inputDim;
    float   epsilon;

    Eigen::Map<const Vector> gamma;
    Eigen::Map<const Vector> beta;


};

template<typename LayerType>
class Residual
{
public:
    explicit Residual(LayerType& _layer) : layer(_layer) {}

    void forward(const Vector& input, Vector& output)
    {
        Vector tmp(input.size());
        layer.forward(input, tmp);
        output = input + tmp;
    }

private:
    LayerType& layer;
};

// Tokenizer class removed - tokenization is handled in executer.cpp


} // namespace layers
