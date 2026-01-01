#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "model.hpp"
#include <nlohmann/json.hpp>
#include <cnpy.h>
#include <unordered_map>
#include "src/model/utils.hpp"

namespace infer {

namespace {

const cnpy::NpyArray& require_array(const cnpy::npz_t& weights, const std::string& name)
{
    auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("Missing array in NPZ: " + name);
    }
    return it->second;
}

Eigen::Map<const MatrixRM> map_matrix(const cnpy::NpyArray& arr,
                                      int expected_rows,
                                      int expected_cols,
                                      const std::string& name)
{
    if (arr.shape.size() == 2) 
    {
        if (expected_rows > 0 && expected_cols > 0) 
        {
            if (arr.shape[0] != static_cast<size_t>(expected_rows) ||
                arr.shape[1] != static_cast<size_t>(expected_cols)) 
                {
                throw std::runtime_error("Unexpected shape for " + name + ": expected [" +
                                         std::to_string(expected_rows) + "," +
                                         std::to_string(expected_cols) + "], got [" +
                                         std::to_string(arr.shape[0]) + "," +
                                         std::to_string(arr.shape[1]) + "]");
                }
        }
        return Eigen::Map<const MatrixRM>(arr.data<float>(),
                                          static_cast<Eigen::Index>(arr.shape[0]),
                                          static_cast<Eigen::Index>(arr.shape[1]));
    }
    if (arr.shape.size() > 2 && arr.shape[0] == 1) 
    {
        size_t cols = 1;
        for (size_t i = 2; i < arr.shape.size(); ++i) 
        {
            cols *= arr.shape[i];
        }
        if (expected_rows > 0 && expected_cols > 0) 
        {
            if (arr.shape[1] != static_cast<size_t>(expected_rows) ||
                cols != static_cast<size_t>(expected_cols)) 
                {
                throw std::runtime_error("Unexpected shape for " + name + ": expected [" +
                                         std::to_string(expected_rows) + "," +
                                         std::to_string(expected_cols) + "], got [" +
                                         std::to_string(arr.shape[1]) + "," +
                                         std::to_string(cols) + "]");
            }
        }
        return Eigen::Map<const MatrixRM>(arr.data<float>(),
                                          static_cast<Eigen::Index>(arr.shape[1]),
                                          static_cast<Eigen::Index>(cols));
    }
    throw std::runtime_error("Expected 2D array for matrix mapping for " + name + ", got size " +
                            std::to_string(arr.shape.size()));
}

Eigen::Map<const Vector> map_vector(const cnpy::NpyArray& arr,
                                    int expected_size,
                                    const std::string& name)
{
    if (arr.shape.size() != 1) 
    {
        throw std::runtime_error("Expected 1D array for vector mapping for " + name + ", got size " +
                                 std::to_string(arr.shape.size()));
    }
    if (expected_size > 0 &&
        arr.shape[0] != static_cast<size_t>(expected_size)) 
        {
        throw std::runtime_error("Unexpected shape for " + name + ": expected [" +
                                 std::to_string(expected_size) + "], got [" +
                                 std::to_string(arr.shape[0]) + "]");
    }
    return Eigen::Map<const Vector>(arr.data<float>(),
                                    static_cast<Eigen::Index>(arr.num_vals));
}

AttentionWeights load_attention(const cnpy::npz_t& weights,
                                const std::string& prefix,
                                int embed_dim)
{
    return AttentionWeights{map_matrix(require_array(weights, prefix + ".Wq.weight"),
                                    embed_dim, embed_dim, prefix + ".Wq.weight"),
                            map_vector(require_array(weights, prefix + ".Wq.bias"),
                                    embed_dim, prefix + ".Wq.bias"),
                            map_matrix(require_array(weights, prefix + ".Wk.weight"),
                                    embed_dim, embed_dim, prefix + ".Wk.weight"),
                            map_vector(require_array(weights, prefix + ".Wk.bias"),
                                    embed_dim, prefix + ".Wk.bias"),
                            map_matrix(require_array(weights, prefix + ".Wv.weight"),
                                    embed_dim, embed_dim, prefix + ".Wv.weight"),
                            map_vector(require_array(weights, prefix + ".Wv.bias"),
                                    embed_dim, prefix + ".Wv.bias"),
                            map_matrix(require_array(weights, prefix + ".Wo.weight"),
                                    embed_dim, embed_dim, prefix + ".Wo.weight"),
                            map_vector(require_array(weights, prefix + ".Wo.bias"),
                                    embed_dim, prefix + ".Wo.bias")
    };
}

FeedForwardWeights load_ff(const cnpy::npz_t& weights,
                           const std::string& prefix,
                           int embed_dim,
                           int ff_dim)
{
    return FeedForwardWeights{
        map_matrix(require_array(weights, prefix + ".fullyConnected1.weight"),
                   ff_dim, embed_dim, prefix + ".fullyConnected1.weight"),
        map_vector(require_array(weights, prefix + ".fullyConnected1.bias"),
                   ff_dim, prefix + ".fullyConnected1.bias"),
        map_matrix(require_array(weights, prefix + ".fullyConnected2.weight"),
                   embed_dim, ff_dim, prefix + ".fullyConnected2.weight"),
        map_vector(require_array(weights, prefix + ".fullyConnected2.bias"),
                   embed_dim, prefix + ".fullyConnected2.bias")
    };
}

ResidualAttention load_residual_attention(const cnpy::npz_t& weights,
                                          const std::string& prefix,
                                          int embed_dim)
{
    return ResidualAttention{
        load_attention(weights, prefix + ".module", embed_dim),
        map_vector(require_array(weights, prefix + ".norm.weight"),
                   embed_dim, prefix + ".norm.weight"),
        map_vector(require_array(weights, prefix + ".norm.bias"),
                   embed_dim, prefix + ".norm.bias")
    };
}

ResidualFeedForward load_residual_ff(const cnpy::npz_t& weights,
                                     const std::string& prefix,
                                     int embed_dim,
                                     int ff_dim)
{
    return ResidualFeedForward{
        load_ff(weights, prefix + ".module", embed_dim, ff_dim),
        map_vector(require_array(weights, prefix + ".norm.weight"),
                   embed_dim, prefix + ".norm.weight"),
        map_vector(require_array(weights, prefix + ".norm.bias"),
                   embed_dim, prefix + ".norm.bias")
    };
}

}  // namespace

Json load_json(const std::string& path) 
{
    std::cout<<"Reading model from the path : "<< path << std::endl;
    std::ifstream handle(path);
    if (!handle.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }
    Json payload;
    handle >> payload;
    return payload;
}

cnpy::npz_t load_npz(const std::string& path) 
{
    std::cout<<"Reading model from the path : "<< path << std::endl;
    return cnpy::npz_load(path);
}

std::unordered_map<std::string, int> load_vocab(const std::string& path) 
{
    std::unordered_map<std::string, int> vocab;
    std::ifstream in(path);
    std::string token;
    int idx = 0;
    while (std::getline(in, token)) 
    {
        vocab[token] = idx++;
    }
    return vocab;
}

// Loading the model weights and mapping to Eigen matrices
ModelWeights load_model_weights(const cnpy::npz_t& weights, const Json& metadata)
{
    ModelWeights model;
    model.named = weights;
    model.encoder.clear();

    const auto& model_cfg = metadata.at("config").at("model");
    const int embed_dim = model_cfg.value("embed_dim", 0);
    const double mlp_ratio = model_cfg.value("mlp_ratio", 0.0);
    const int max_length = model_cfg.value("max_length", 0);
    const int vocab_size = model_cfg.value("vocab_size", 0);
    const int ff_dim = (embed_dim > 0 && mlp_ratio > 0.0)
        ? static_cast<int>(std::lround(embed_dim * mlp_ratio))
        : 0;

    std::cout << "Model embed_dim: " << embed_dim << std::endl;
    std::cout << "Model ff_dim: " << ff_dim << std::endl;

    std::cout<< "Reading model weights.."<< std::endl;
    for (size_t i = 0; ; ++i) 
    {
        const std::string layer_prefix = "encoder." + std::to_string(i);

        if (weights.find(layer_prefix + ".attention.Wq.weight") == weights.end()) 
        {
            break;
        }

        model.encoder.push_back(EncoderLayerWeights{load_attention(weights, layer_prefix + ".attention", embed_dim),
                                                    load_ff(weights, layer_prefix + ".ff", embed_dim, ff_dim),
                                                    load_residual_attention(weights, layer_prefix + ".residue1", embed_dim),
                                                    load_residual_ff(weights, layer_prefix + ".residue2", embed_dim, ff_dim)
                                                });
    }

    auto set_optional_matrix = [&](std::unique_ptr<Eigen::Map<const MatrixRM>>& target,
                                   const std::string& key,
                                   int expected_rows,
                                   int expected_cols) {
        auto it = weights.find(key);
        if (it != weights.end()) {
            target = std::make_unique<Eigen::Map<const MatrixRM>>(
                map_matrix(it->second, expected_rows, expected_cols, key));
        }
    };

    auto set_optional_vector = [&](std::unique_ptr<Eigen::Map<const Vector>>& target,
                                   const std::string& key,
                                   int expected_size) {
        auto it = weights.find(key);
        if (it != weights.end()) {
            target = std::make_unique<Eigen::Map<const Vector>>(
                map_vector(it->second, expected_size, key));
        }
    };

    const int expected_vocab = vocab_size > 0 ? vocab_size : 0;
    const int expected_positions = max_length > 0 ? max_length + 1 : 0;

    set_optional_matrix(model.cls_token, "cls_token",
                        embed_dim > 0 ? 1 : 0, embed_dim);
    set_optional_matrix(model.token_embedding_weight, "token_embedding.weight",
                        expected_vocab, embed_dim);
    set_optional_matrix(model.position_positional_table, "position.positional_table",
                        expected_positions, embed_dim);
    set_optional_vector(model.norm_weight, "norm.weight", embed_dim);
    set_optional_vector(model.norm_bias, "norm.bias", embed_dim);
    set_optional_matrix(model.head0_weight, "head.0.weight", embed_dim, embed_dim);
    set_optional_vector(model.head0_bias, "head.0.bias", embed_dim);
    set_optional_matrix(model.head3_weight, "head.3.weight", 1, embed_dim);
    set_optional_vector(model.head3_bias, "head.3.bias", 1);

    return model;
}

std::vector<std::string> find_keys_with_prefix(const ModelWeights& weights,
                                               const std::string& prefix)
{
    std::vector<std::string> matches;
    for (const auto& kv : weights.named) {
        if (kv.first.rfind(prefix, 0) == 0) {
            matches.push_back(kv.first);
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}

const cnpy::NpyArray* find_array(const ModelWeights& weights,
                                 const std::string& key)
{
    auto it = weights.named.find(key);
    if (it == weights.named.end()) {
        return nullptr;
    }
    return &it->second;
}

LoadedParams load_params(const std::string& json_path, const std::string& npz_path, const std::string& vocab_path) 
{
    LoadedParams params;
    params.metadata      = load_json(json_path);
    params.weights       = load_npz(npz_path);

    params.model_weights = load_model_weights(params.weights, params.metadata);
    params.vocab         = load_vocab(vocab_path);

    return params;
}

}  // namespace infer
