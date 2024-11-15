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
    cv::Mat image = cv::imread(image_path);

    // Check if the image was loaded successfully
    if (image.empty()) {
        throw runtime_error("Error: Could not load image from " + image_path);
    }

    // Convert image into a tensor that can be used by the CLIP model
    torch::Tensor image_input = preprocess_image(image).to(torch::kCUDA);

    // Get attention map
    torch::Tensor attn_map = get_attention_map(image_input, model).to(torch::kCUDA);

    // Get edge map
    torch::Tensor edge_map = get_edge_map(image).to(torch::kCUDA);

    // Get distribution map by multiplying the attention map with the edge map
    torch::Tensor distribution_map = attn_map * edge_map;
    // Softmax only values above zero
    distribution_map = softmax_above_zero(distribution_map);




    cv::Mat temp(224, 224, CV_32FC1, edge_map.to(torch::kCPU).data_ptr<float>());
    temp.convertTo(temp, CV_8UC1, 255);
    cv::imwrite("../temp.png", temp);

    cv::Mat temp2(224, 224, CV_32FC1, attn_map.to(torch::kCPU).data_ptr<float>());
    temp2.convertTo(temp2, CV_8UC1, 255);
    cv::imwrite("../temp2.png", temp2);

    cv::Mat temp3(224, 224, CV_32FC1, distribution_map.to(torch::kCPU).data_ptr<float>());
    temp3.convertTo(temp3, CV_8UC1, 255);
    cv::imwrite("../temp3.png", temp3);
}

// Function to preprocess the image
torch::Tensor preprocess_image(cv::Mat image_input) {
    cv::Mat image = image_input.clone();

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
    torch::Tensor img_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).unsqueeze(0);

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
torch::Tensor get_attention_map(torch::Tensor image_input, torch::jit::script::Module model){
    torch::Tensor image = image_input.clone();

    // Encode image to fill model with values (result is not needed further)
    auto result = model.run_method("encode_image", image);

    // Access the 12 ResidualAttentionBlocks
    auto resblocks = model.attr("visual").toModule().attr("transformer").toModule().attr("resblocks").toModule();

    // Store the 12 attention maps in a vector
    vector<torch::Tensor> attns;
    for (const auto& named_child : resblocks.named_children()) {
        torch::Tensor attn = named_child.value.attr("attn_weights").toTensor().detach(); // [1, 50, 50]
        attns.push_back(attn);
    }

    torch::Tensor attn_map;
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
    
    // Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min()); 

    return attn_map;
}

// Get the edgemap to the given image using XDoG edge detection
torch::Tensor get_edge_map(cv::Mat image_input) {
    double gamma = 0.98;
    double phi = 200.0;
    double epsilon = -0.1;
    double sigma = 0.8;
    double k = 10;

    cv::Mat image = image_input.clone();

    // Grayscale image
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    cv::Mat blurred1, blurred2;
    cv::GaussianBlur(image, blurred1, cv::Size(0, 0), sigma);
    cv::GaussianBlur(image, blurred2, cv::Size(0, 0), sigma * k);

    cv::Mat xdog = blurred1 - (gamma * blurred2);

    // Convert to Float32 and normalize
    xdog.convertTo(xdog, CV_32F, 1.0 / 255.0);
    
    for (int y = 0; y < xdog.rows; y++) {
        for (int x = 0; x < xdog.cols; x++) {
            float val = xdog.at<float>(y, x);

            if (val < epsilon) {
                xdog.at<float>(y, x) = 1.0f;
            } else {
                xdog.at<float>(y, x) = 1.0f + tanh(phi * (val + epsilon));
            }
        }
    }

    // Resize
    cv::resize(xdog, xdog, cv::Size(224, 224), (0,0), (0,0), cv::INTER_CUBIC);
    
    // CenterCrop image
    int x = (xdog.cols - 224) / 2; // Set starting points
    int y = (xdog.rows - 224) / 2; // Set starting points
    cv::Rect crop_region(x, y, 224, 224); // Specify crop region
    xdog = xdog(crop_region); // Crop image

    // Convert image to tensor
    torch::Tensor xdog_tensor = torch::from_blob(xdog.data, {xdog.rows, xdog.cols}, torch::kFloat32);

    // Normalize
    xdog_tensor = (xdog_tensor - xdog_tensor.min()) / (xdog_tensor.max() - xdog_tensor.min());

    // Binarize
    xdog_tensor = (xdog_tensor >= 0.5f).to(torch::kFloat32);

    return xdog_tensor;
}

// Softmax given tensor but only consider values above zero
torch::Tensor softmax_above_zero(torch::Tensor x) {
    x = torch::where(x > 0, x, -std::numeric_limits<float>::infinity()); // set all values that are 0 to negative infinity
    torch::Tensor e_x = torch::exp(x - x.max());
    return e_x / e_x.sum();
}