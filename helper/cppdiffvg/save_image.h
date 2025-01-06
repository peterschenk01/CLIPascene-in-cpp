#ifndef SAVESVG_H_
#define SAVESVG_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "tinyxml2.h"
#include "path.h"

void save_svg(const string filename,
              int width,
              int height,
              vector<PathCPP> paths
              );

void save_png(const string filename, torch::Tensor img);

#endif