#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <stdexcept>
#include "executer.hpp"

#ifdef USE_TOKENIZERS
#include <tokenizers_cpp.h>
#endif


namespace execute
{

    Executer::Executer(const infer::LoadedParams& _modelParams)
    : modelParams(_modelParams) {}

    std::string Executer::readInput()
    {
        std::string input;
        std::getline(std::cin, input);
        return input;
    }

    void Executer::run()
    {
        std::cout << "Enter the input: " << std::endl;
        std::string input = readInput();
        std::cout << "Input: " << input << std::endl;

#ifdef USE_TOKENIZERS
        if (modelParams.tokenizer) {
            // Use tokenizers-cpp API: Encode returns std::vector<int>
            std::vector<int> token_ids = modelParams.tokenizer->Encode(input);

            std::cout << "Token IDs (" << token_ids.size() << " tokens):";
            for (auto id : token_ids) {
                std::cout << " " << id;
            }
            std::cout << std::endl;

            // Demonstrate decoding
            std::string decoded = modelParams.tokenizer->Decode(token_ids);
            std::cout << "Decoded: " << decoded << std::endl;

            // TODO: Implement inference pipeline here
            // 1. Convert token IDs to embeddings
            // 2. Add positional encoding
            // 3. Pass through encoder layers
            // 4. Apply classification head
            // 5. Return predictions

        } else {
            std::cerr << "Error: Tokenizer not loaded!" << std::endl;
        }
#else
        std::cout << "Warning: Tokenizer support not enabled (compile with -DUSE_TOKENIZERS=ON)" << std::endl;
        std::cout << "Using vocab map for basic tokenization..." << std::endl;

        // Fallback: simple whitespace tokenization with vocab lookup
        std::istringstream iss(input);
        std::string token;
        std::vector<int> token_ids;

        while (iss >> token) {
            auto it = modelParams.vocab.find(token);
            if (it != modelParams.vocab.end()) {
                token_ids.push_back(it->second);
            } else {
                // Try to find [UNK] token
                auto unk_it = modelParams.vocab.find("[UNK]");
                if (unk_it != modelParams.vocab.end()) {
                    token_ids.push_back(unk_it->second);
                }
            }
        }

        std::cout << "Token IDs (" << token_ids.size() << " tokens):";
        for (auto id : token_ids) {
            std::cout << " " << id;
        }
        std::cout << std::endl;
#endif
    }
    
} // namespace execute
