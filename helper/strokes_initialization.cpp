#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "strokes_initialization.h"

void strokes_initialization() {
    std::string model_path = "../models/clip_vit_b32.pt";
    std::string image_path = "../input_images/input_image.png";
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

    // Load image
    cv::Mat image = cv::imread(image_path);

    // Check if the image was loaded successfully
    if (image.empty()) {
        throw std::runtime_error("Error: Could not load image from " + image_path);
    }

    torch::Tensor tensor = preprocess_image(image);
    torch::save(tensor, "../cpp_tensor.pt");
}

// Function to preprocess the image
torch::Tensor preprocess_image(const cv::Mat img) {
    cv::Mat image = img;

    // Resize to 224x224
    cv::resize(image, image, cv::Size(224, 224), (0,0), (0,0), cv::INTER_CUBIC);
    
    // CenterCrop image
    image = center_crop(image, 224);

    // Convert image to RGB
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Image is in BGR format, convert to RGB
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGB); // Image is in BGRA format, convert to RGB
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB); // If grayscale, convert to RGB by replicating the single channel
    }

    // Convert image to tensor
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
    img_tensor = img_tensor.to(torch::kFloat) / 255.0;

    // Permute to (C, H, W) format
    img_tensor = img_tensor.permute({0, 3, 1, 2});

    // Normalize
    img_tensor = torch::data::transforms::Normalize<>(
        {0.48145466, 0.4578275, 0.40821073},  // Mean for CLIP
        {0.26862954, 0.26130258, 0.27577711}  // Std deviation for CLIP
    )(img_tensor);

    return img_tensor.clone(); // Clone to avoid memory issues
}


// CenterCrop function: crops the central region of an image
cv::Mat center_crop(const cv::Mat &image, int size) {
    // Set starting points
    int x = (image.cols - size) / 2;
    int y = (image.rows - size) / 2;
    
    // Specify crop region
    cv::Rect crop_region(x, y, size, size);

    // Crop and return the image
    return image(crop_region).clone();
}
