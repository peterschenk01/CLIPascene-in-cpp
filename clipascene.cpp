#include <iostream>
#include <torch/torch.h>
#include "helper/strokes_initialization.h"

int main(int argc, char *argv[]) {
    
    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Exiting..." << std::endl;
        return -1;
    }

    strokes_initialization();
    
    return 0;
}