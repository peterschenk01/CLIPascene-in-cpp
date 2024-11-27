#include <iostream>
#include <cassert>
#include <torch/torch.h>
#include "cppdiffvg.h"
#include "diffvg.h"
#include "filter.h"
#include "pixel_filter.h"
#include "path.h"
#include "shape.h"

using namespace std;
using custom_variant = variant<int, bool, torch::Tensor, OutputType, ShapeType, ColorType, FilterType>;

/*
In this file I implemented the RenderFunction from the Pytorch part of diffvg in C++.
Since I only need the ShapeType Path I ignored everything that had to do with other ShapeTypes.
I also changed a couple of things if I didn't need their functionality.
You can find the original function in render_pytorch.py in the diffvg GitHub.
*/


vector<custom_variant> RenderFunction::serialize_scene(int canvas_width,
                                                    int canvas_height,
                                                    vector<path::Path> paths,
                                                    vector<path::PathGroup> path_groups,
                                                    PixelFilter filter,
                                                    OutputType output_type,
                                                    bool use_prefiltering,
                                                    torch::Tensor eval_positions
                                                    ) {

    int num_shapes = static_cast<int>(paths.size());
    int num_shape_groups = static_cast<int>(path_groups.size());

    vector<custom_variant> args;

    args.push_back(canvas_width);
    args.push_back(canvas_height);
    args.push_back(num_shapes);
    args.push_back(num_shape_groups);
    args.push_back(output_type);
    args.push_back(use_prefiltering);
    args.push_back(eval_positions.to(torch::kCUDA));

    for(path::Path pth : paths) {
        bool use_thickness = false;
        assert(pth.num_control_points.is_contiguous());
        assert(pth.points.is_contiguous());
        assert(pth.points.size({1}) == 2);
        assert(torch::isfinite(pth.points).all().item<bool>());
        args.push_back(ShapeType::Path);
        args.push_back(pth.num_control_points.to(torch::kInt32).cpu());
        args.push_back(pth.points.cpu());
        if((pth.stroke_width.sizes().size() > 0) && (pth.stroke_width.sizes()[0] > 1)) {
            assert(torch::isfinite(pth.stroke_width).all().item<bool>());
            use_thickness = true;
            args.push_back(pth.stroke_width.cpu());
        } else {
            args.push_back(false);
        }
        args.push_back(pth.is_closed);
        args.push_back(pth.use_distance_approx);
        if(use_thickness) {
            args.push_back(use_thickness);
        } else {
            args.push_back(pth.stroke_width.cpu());
        }
    }

    for(path::PathGroup grp : path_groups) {
        assert(grp.shape_ids.is_contiguous());
        args.push_back(grp.shape_ids.to(torch::kInt32).cpu());
        args.push_back(false); // No fill color

        assert(grp.stroke_color.is_contiguous());
        args.push_back(ColorType::Constant);
        args.push_back(grp.stroke_color.cpu());

        args.push_back(grp.use_even_odd_rule);
        args.push_back(grp.shape_to_canvas.contiguous().cpu());
    }

    args.push_back(filter.type);
    args.push_back(filter.radius.cpu());

    return args;
}

// Forward rendering pass
torch::Tensor RenderFunction::forward(torch::autograd::AutogradContext *ctx,
                                    int width,
                                    int height,
                                    int num_samples_x,
                                    int num_samples_y,
                                    int seed,
                                    torch::Tensor background_image,
                                    vector<custom_variant> *args
                                    ) {
    
    // Unpack arguments
    int current_index = 0;
    auto canvas_width = args[current_index];
    current_index += 1;
    auto canvas_height = args[current_index];
    current_index += 1;
    auto num_shapes = args[current_index];
    current_index += 1;
    auto num_shape_groups = args[current_index];
    current_index += 1;
    auto output_type = args[current_index];
    current_index += 1;
    auto use_prefiltering = args[current_index];
    current_index += 1;
    auto eval_positions = args[current_index];
    current_index += 1;

    return torch::Tensor();
}