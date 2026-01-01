#include <iostream>
#include <string>
#include "model.hpp"

int main(int argc, char** argv) 
{
    bool display_params = false;
    for (int i = 1; i < argc; ++i) 
    {
        const std::string arg = argv[i];
        if (arg == "--display-params=true") 
        {
        display_params = true;
        } 
        else if (arg == "--display-params=false") 
        {
        display_params = false;
        }
    }

    infer::LoadedParams modelParams;

    modelParams = infer::load_params("inference/exports/imdb_checkpoint.json",
                                     "inference/exports/imdb_checkpoint.npz",
                                     "inference/exports/bert_tokenizer/vocab.txt");
    
    if(display_params)
    {
        const auto& config = modelParams.metadata;

        std::cout << config.at("config").at("model") << std::endl;

        const auto& weights = modelParams.weights;

        for (const auto& kv : weights) 
        {
            const std::string& name = kv.first;
            const cnpy::NpyArray& arr = kv.second;

            std::cout << "array: " << name << " shape=[";
            for (size_t i = 0; i < arr.shape.size(); ++i) 
            {
                std::cout << arr.shape[i] << (i + 1 < arr.shape.size() ? "," : "");
            }
            std::cout << "] word_size=" << arr.word_size << std::endl;

            // Print first few values (assumes float32)
            const float* data = arr.data<float>();
            size_t count = std::min<size_t>(arr.num_vals, 8);
            std::cout << "  first values:";
            for (size_t i = 0; i < count; ++i) 
            {
                std::cout << " " << data[i];
            }
            std::cout << std::endl;
        }
    
        std::cout << modelParams.vocab["[PAD]"] << std::endl;
    }

    std::cout << "tinystories_app: placeholder main (wiring pending)" << std::endl;
    return 0;
}
