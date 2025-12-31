#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "utils.hpp"
#include "layers.hpp"
#include <cmath>



namespace layers
{

 Linear::Linear(const int32_t _inputDim,
                const int32_t _outputDim,
                const float_t* wPtr,
                const float_t* bPtr,
                const std::string _name)

          : inputDim(std::move(_inputDim)),
            outputDim(std::move(_outputDim)),
            name(std::move(_name)),
            weights(wPtr, inputDim, outputDim),
            bias(bPtr, outputDim) {}


    void Linear::forward(const RowVec& input, RowVec& output)
    {
        output.noalias() = input * weights;
        output          += bias.transpose();

    }


 LayerNorm::LayerNorm(const float* g_ptr,
                      const float* b_ptr,
                      int d,
                      float eps_)

                : inputDim(d),
                  epsilon(eps_),
                  gamma(g_ptr, d),
                  beta(b_ptr, d) {}

   void LayerNorm::forward(const Vector& input,  Vector& ouput)
    {
        float mean  = input.mean();
        float var   = (input.array() - mean).square().mean();
        float inv   = 1.0f / std::sqrt(var + epsilon);
        ouput       = (input.array() - mean) * inv;
        ouput       = ouput.array() * gamma.array() + beta.array();
    }
}
