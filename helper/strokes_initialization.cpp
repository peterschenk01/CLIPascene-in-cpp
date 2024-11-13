#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "strokes_initialization.h"

using namespace std;

void strokes_initialization() {
    string model_path = "../models/clip_vit_b32_scripted.pt";
    string image_path = "../input_images/input_image.png";
    torch::jit::script::Module model;

    // Load the model
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        cerr << "Error loading the model: " << e.what() << endl;
        return;
    }

    // Move model to GPU
    model.to(torch::kCUDA);

    cout << "Successfully loaded the model and moved to GPU" << endl;

    // Load image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);

    // Check if the image was loaded successfully
    if (image.empty()) {
        throw runtime_error("Error: Could not load image from " + image_path);
    }

    // Convert image into a tensor that can be used by the CLIP model
    at::Tensor image_input = preprocess_image(image).to(torch::kCUDA);

    // Get attention map
    at::Tensor attn_map = get_attention_map(image_input, model);

    cout << attn_map << endl;

    torch::save(attn_map.cpu(), "../cpp_tensor.pt");
}

// Function to preprocess the image
at::Tensor preprocess_image(const cv::Mat image_input) {
    cv::Mat image = image_input;

    // Convert image to RGB
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Image is in BGR format, convert to RGB
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGB); // Image is in BGRA format, convert to RGB
    } else if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB); // If grayscale, convert to RGB by replicating the single channel
    } else {
        throw invalid_argument("Unsupported image format: must be 1, 3, or 4 channels.");
    }

    // Resize to 224x224
    cv::resize(image, image, cv::Size(224, 224), (0,0), (0,0), cv::INTER_CUBIC);
    
    // CenterCrop image
    int x = (image.cols - 224) / 2; // Set starting points
    int y = (image.rows - 224) / 2; // Set starting points
    cv::Rect crop_region(x, y, 224, 224); // Specify crop region
    image = image(crop_region); // Crop image

    // Convert image to tensor
    at::Tensor img_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).unsqueeze(0);

    // Convert to Float32 and Normalize
    img_tensor = img_tensor.toType(torch::kFloat32).div(255);

    // Permute to [B, C, H, W] Format
    img_tensor = img_tensor.permute({0, 3, 1, 2});

    // Normalize
    img_tensor = torch::data::transforms::Normalize<>(
        {0.48145466, 0.4578275, 0.40821073},  // Mean for CLIP
        {0.26862954, 0.26130258, 0.27577711}  // Std deviation for CLIP
    )(img_tensor);

    return img_tensor;
}

// Get the attention map from the given image
at::Tensor get_attention_map(at::Tensor image_input, torch::jit::script::Module model){
    at::Tensor image = image_input.clone();

    // Encode image to fill model with values (result is not needed further)
    auto result = model.run_method("encode_image", image);

    // Access the 12 ResidualAttentionBlocks
    auto resblocks = model.attr("visual").toModule().attr("transformer").toModule().attr("resblocks").toModule();

    // Store the 12 attention maps in a vector
    vector<at::Tensor> attns;
    for (const auto& named_child : resblocks.named_children()) {
        at::Tensor attn = named_child.value.attr("attn_weights").toTensor().detach(); // [1, 50, 50]
        attns.push_back(attn);
    }

    at::Tensor attn_map;
    attn_map = torch::cat(attns); // Concatenate into single Tensor [12, 50, 50]
    attn_map = attn_map.index({torch::indexing::Slice(), 0, torch::indexing::Slice(1, torch::indexing::None)}); // [12, 1, 49]
    attn_map = attn_map.mean(0).unsqueeze(0); // Average the 12 maps
    attn_map = attn_map.reshape({1, 1, 7, 7});
    attn_map = torch::nn::functional::interpolate(
        attn_map, 
        torch::nn::functional::InterpolateFuncOptions().size(vector<int64_t>{224, 224}).mode(torch::kBicubic).align_corners(false)
        );
    attn_map = attn_map.reshape({224, 224});
    attn_map = attn_map.toType(torch::kFloat32);
    
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min()); // Normalize

    return attn_map;
}