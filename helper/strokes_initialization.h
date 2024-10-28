#ifndef STROKES_INITIALIZATION_H
#define STROKES_INITIALIZATION_H

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

void strokes_initialization();

torch::Tensor preprocess_image(const cv::Mat img);

#endif