#ifndef STROKES_INITIALIZATION_H
#define STROKES_INITIALIZATION_H

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

void strokes_initialization();

at::Tensor preprocess_image(const cv::Mat img);

at::Tensor get_attention_map(at::Tensor image_input, torch::jit::Module model);

#endif