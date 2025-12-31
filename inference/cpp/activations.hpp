#include <stdexcept>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utils.hpp"
#include <cmath>

namespace activations
{

class softmax
{
    softmax() = default;

    void forward(const Vector& input, Vector& output) const;

};

class Softmax1D 
{

    Softmax1D() = default;

    // In-place softmax on x[0..n-1]
    void forward_inplace(float* x, int n) const;
};

}
