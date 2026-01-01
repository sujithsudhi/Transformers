#pragma once

#include <nlohmann/json.hpp>
#include <cnpy.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "src/model/utils.hpp"

namespace infer {

using Json = nlohmann::json;

struct AttentionWeights
{
    Eigen::Map<const MatrixRM> Wq_weight;
    Eigen::Map<const Vector> Wq_bias;
    Eigen::Map<const MatrixRM> Wk_weight;
    Eigen::Map<const Vector> Wk_bias;
    Eigen::Map<const MatrixRM> Wv_weight;
    Eigen::Map<const Vector> Wv_bias;
    Eigen::Map<const MatrixRM> Wo_weight;
    Eigen::Map<const Vector> Wo_bias;
};

struct FeedForwardWeights
{
    Eigen::Map<const MatrixRM> fc1_weight;
    Eigen::Map<const Vector> fc1_bias;

    Eigen::Map<const MatrixRM> fc2_weight;
    Eigen::Map<const Vector> fc2_bias;
};

struct ResidualAttention
{
    AttentionWeights module;
    Eigen::Map<const Vector> norm_weight;
    Eigen::Map<const Vector> norm_bias;
};

struct ResidualFeedForward
{
    FeedForwardWeights module;
    Eigen::Map<const Vector> norm_weight;
    Eigen::Map<const Vector> norm_bias;
};

struct EncoderLayerWeights
{
    AttentionWeights attention;
    FeedForwardWeights ff;
    ResidualAttention residue1;
    ResidualFeedForward residue2;
};

struct ModelWeights
{
    std::map<std::string, cnpy::NpyArray> named;
    std::vector<EncoderLayerWeights> encoder;

    std::unique_ptr<Eigen::Map<const MatrixRM>> cls_token;
    std::unique_ptr<Eigen::Map<const MatrixRM>> token_embedding_weight;
    std::unique_ptr<Eigen::Map<const MatrixRM>> position_positional_table;
    std::unique_ptr<Eigen::Map<const Vector>> norm_weight;
    std::unique_ptr<Eigen::Map<const Vector>> norm_bias;

    std::unique_ptr<Eigen::Map<const MatrixRM>> head0_weight;
    std::unique_ptr<Eigen::Map<const Vector>> head0_bias;

    std::unique_ptr<Eigen::Map<const MatrixRM>> head3_weight;
    std::unique_ptr<Eigen::Map<const Vector>> head3_bias;
};

struct LoadedParams 
{
    Json metadata;
    cnpy::npz_t weights;
    ModelWeights model_weights;
    std::unordered_map<std::string, int> vocab;
};

// Loading the model parameters
LoadedParams load_params(const std::string& json_path, 
                         const std::string& npz_path, 
                         const std::string& vocab_path);

// Loading model weights
cnpy::npz_t load_npz(const std::string& path);

// Loading model config
Json load_json(const std::string& path);

ModelWeights load_model_weights(const cnpy::npz_t& weights, const Json& metadata);

std::unordered_map<std::string, int> load_vocab(const std::string& path);

std::vector<std::string> find_keys_with_prefix(const ModelWeights& weights,
                                               const std::string& prefix);
const cnpy::NpyArray* find_array(const ModelWeights& weights,
                                 const std::string& key);

}  // namespace infer
