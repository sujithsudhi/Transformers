#include <stdexcept>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utils.hpp"
#include <cmath>


namespace layers
{
    // Here we define the Linear layer (Dense layer without activation)
class Linear
{
    
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

class Residual
{
    Residual()
}


} // namespace layer
