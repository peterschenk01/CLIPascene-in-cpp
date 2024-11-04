#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "strokes_initialization.h"

void strokes_initialization() {
    std::string model_path = "../models/clip_vit_b32_scripted.pt";
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

    // Convert image into a tensor that can be used by the CLIP model
    torch::Tensor image_input = preprocess_image(image).to(torch::kCUDA);
    // torch::save(image_input, "../cpp_tensor.pt");

    attn_map(image_input, model);
}

// Function to preprocess the image
torch::Tensor preprocess_image(const cv::Mat img) {
    cv::Mat image = img;

    // Resize to 224x224
    cv::resize(image, image, cv::Size(224, 224), (0,0), (0,0), cv::INTER_CUBIC);
    
    // CenterCrop image
    int x = (image.cols - 224) / 2; // Set starting points
    int y = (image.rows - 224) / 2; // Set starting points
    cv::Rect crop_region(x, y, 224, 224); // Specify crop region
    image = image(crop_region); // Crop image

    // Convert image to RGB
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Image is in BGR format, convert to RGB
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGB); // Image is in BGRA format, convert to RGB
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB); // If grayscale, convert to RGB by replicating the single channel
    } else {
        throw std::invalid_argument("Unsupported image format: must be 1, 3, or 4 channels.");
    }

    // Convert image to tensor
    torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
    img_tensor = img_tensor.to(torch::kFloat) / 255.0;

    // Permute to (B, C, H, W) format
    img_tensor = img_tensor.permute({0, 3, 1, 2});

    // Normalize
    img_tensor = torch::data::transforms::Normalize<>(
        {0.48145466, 0.4578275, 0.40821073},  // Mean for CLIP
        {0.26862954, 0.26130258, 0.27577711}  // Std deviation for CLIP
    )(img_tensor);

    return img_tensor;
}

void attn_map(torch::Tensor image_input, torch::jit::script::Module model){
    torch::Tensor images = image_input.repeat({1, 1, 1, 1});
    auto res = model.run_method("encode_image", images);

    // Access the resblocks module
    auto resblocks = model.attr("visual").toModule().attr("transformer").toModule().attr("resblocks").toModule();
    // Store each child module in a vector
    std::vector<torch::jit::Module> image_attn_blocks;
    for (const auto& named_child : resblocks.named_children()) {
        image_attn_blocks.push_back(named_child.value);
    }

    auto num_tokens = image_attn_blocks[0].attr("attn").type();

    std::cout << num_tokens->str() << std::endl;

    for (const auto& named_child : resblocks.named_children()){
        std::cout << named_child.name << std::endl;
    }
}