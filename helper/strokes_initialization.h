#ifndef STROKES_INITIALIZATION_H_
#define STROKES_INITIALIZATION_H_

#include <iostream>
#include <random>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "cppdiffvg/cppdiffvg.h"
#include "cppdiffvg/path.h"

void strokes_initialization();

torch::Tensor preprocess_image(cv::Mat image_input);

torch::Tensor get_attention_map(torch::Tensor image_input, torch::jit::script::Module model);

torch::Tensor get_edge_map(cv::Mat image_input);

torch::Tensor softmax_above_zero(torch::Tensor x);

PathCPP get_path(torch::Tensor indices, int strokes_counter);

void visualize_distr_map(torch::Tensor distribution_map, torch::Tensor row_indices, torch::Tensor col_indices, int num_strokes);

#endif