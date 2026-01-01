#include <vector>
#include <string>
#include <map>
#include <iostream>
#include "cnpy.h"
#include <nlohmann/json.hpp>


class ModelConfig 
{
    int n_layer;
    int n_embd;
    int n_head;
    int vocab_size;
    int block_size; // Max context length
};

struct Weights
{


}

struct Parameters
{
    ModelConfig config;

    /* data */
};
