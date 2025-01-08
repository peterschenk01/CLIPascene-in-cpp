#ifndef PATH_H_
#define PATH_H_

#include <iostream>
#include <torch/torch.h>

using namespace std;

/* C++ partial implementation of shape.py from pydiffvg. */

struct PathCPP {
    // Constructor
    PathCPP(
        torch::Tensor num_control_points,
        torch::Tensor points,
        bool is_closed,
        torch::Tensor stroke_width = torch::tensor(1.0),
        string id = "",
        bool use_distance_approx = false
    ) : num_control_points(num_control_points), 
        points(points), 
        is_closed(is_closed),
        stroke_width(stroke_width),
        id(id),
        use_distance_approx(use_distance_approx) {}

    // Member variables
    torch::Tensor num_control_points;
    torch::Tensor points;
    bool is_closed;
    torch::Tensor stroke_width;
    string id;
    bool use_distance_approx;
};

struct PathGroupCPP {
    // Constructor
    PathGroupCPP(
        torch::Tensor shape_ids,
        bool use_even_odd_rule = true,
        torch::Tensor stroke_color = torch::tensor({0.0, 0.0, 0.0, 1.0}),
        torch::Tensor shape_to_canvas = torch::eye(3),
        string id = ""
    ) : shape_ids(shape_ids), 
        use_even_odd_rule(use_even_odd_rule), 
        stroke_color(stroke_color), 
        shape_to_canvas(shape_to_canvas), 
        id(id) {}
    
    // Member variables
    torch::Tensor shape_ids;
    bool use_even_odd_rule;
    torch::Tensor stroke_color;
    torch::Tensor shape_to_canvas;
    string id;
};

#endif