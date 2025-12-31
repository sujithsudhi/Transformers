#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>

using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector   = Eigen::VectorXf;
using RowVec   = Eigen::RowVectorXf;


struct Tensor
{
    std::vector<float> data;
    std::vector<int> shape;
    std::string name;
    
    Tensor(std::vector<float> data_,
           std::vector<int> shape_,
           std::string name_ = "")

        : data(std::move(data_)),
          shape(std::move(shape_)),
          name(std::move(name_))
    {
        size_t expected = 1;
        for (int dim : shape) 
        {
            if (dim <= 0) 
            {
                throw std::invalid_argument("Tensor shape must be positive.");
            }
            expected *= static_cast<size_t>(dim);
        }
        if (expected != data.size()) 
        {
            throw std::invalid_argument("Tensor data size does not match shape.");
        }
    }

    const std::vector<float>& getData() const
    {
        return data;
    }

    const std::vector<int> getShape() const
    {
        return shape;
    }

    const std::string& getName() const
    {
        return name;
    }

};
