#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "strokes_initialization.h"

void strokes_initialization() {
    std::string model_path = "../models/clip_vit_b32.pt";
    std::string image_path = "";
    torch::jit::script::Module model;

    // Load the model
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return;
    }

    // Move model to GPU
    model.to(torch::kCUDA);

    std::cout << "Successfully loaded the model and moved to GPU" << std::endl;
}

// TO DO
// Function to preprocess the image
torch::Tensor preprocess_image(const std::string& image_path) {
    // Load image
    cv::Mat img = cv::imread(image_path);
    cv::resize(img, img, cv::Size(224, 224)); // Resize to 224x224

    // Convert to float and normalize
    img.convertTo(img, CV_32F, 1.0 / 255);
    auto img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kFloat);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // Change to CxHxW

    return img_tensor.clone(); // Clone to avoid memory issues
}
