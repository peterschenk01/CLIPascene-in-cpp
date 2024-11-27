#ifndef CPPDIFFVG_H_
#define CPPDIFFVG_H_

#include <iostream>
#include <cassert>
#include <torch/torch.h>
#include "diffvg.h"
#include "filter.h"
#include "pixel_filter.h"
#include "path.h"
#include "shape.h"

using namespace std;

enum class OutputType {
    color = 1,
    sdf = 2
};

class RenderFunction : public torch::autograd::Function<RenderFunction> {
public:
    using custom_variant = variant<int, bool, torch::Tensor, OutputType, ShapeType, ColorType, FilterType>;

    static vector<custom_variant> serialize_scene(int canvas_width,
                                                int canvas_height,
                                                vector<path::Path> paths,
                                                vector<path::PathGroup> path_groups,
                                                PixelFilter filter = PixelFilter(FilterType::Box, torch::tensor(0.5)),
                                                OutputType output_type = OutputType::color,
                                                bool use_prefiltering = false,
                                                torch::Tensor eval_positions = torch::tensor({})
                                                );

    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                int width,
                                int height,
                                int num_samples_x,
                                int num_samples_y,
                                int seed,
                                torch::Tensor background_image,
                                vector<custom_variant> *args
                                );

};

#endif