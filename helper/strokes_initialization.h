#include <iostream>
#include <random>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

void strokes_initialization();

torch::Tensor preprocess_image(cv::Mat image_input);

torch::Tensor get_attention_map(torch::Tensor image_input, torch::jit::script::Module model);

torch::Tensor get_edge_map(cv::Mat image_input);

torch::Tensor softmax_above_zero(torch::Tensor x);

void get_path(torch::Tensor indices, int strokes_counter);