#include <vector>
#include <string>
#include <map>
#include <iostream>
#include "cnpy.h"
#include <nlohmann/json.hpp>
#include "model.hpp"

namespace execute {


class Executer
{
    public:

        // Main class for the inference execution
        explicit Executer(const infer::LoadedParams& _modelParams);

        // Read the inference
        std::string readInput();
        
        // Run the inference
        void run();

    private:
        const infer::LoadedParams& modelParams;
        

};

}
