#include "utils.hpp"
#include "activations.hpp"
#include <cmath>

namespace activations
{
    void softmax::forward(const Vector& input, Vector& output) const
    {
        // get the maximum value
        float maximum = input.maxCoeff();

        // calculates elementwise exponentials
        output        = (input.array() - maximum).exp();

        // getting the total sum of the exponential
        float sum     = output.sum();

        // performing elementwise division
        output       /= sum;
    }

    void Softmax1D::forward_inplace(float* x, int n) const 
    {
        // max
        float mx = x[0];
        for (int i = 1; i < n; i++) mx = std::max(mx, x[i]);

        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < n; i++) 
        {
            x[i] = std::exp(x[i] - mx);
            sum += x[i];
        }

        // normalize
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) x[i] *= inv;
    }
}