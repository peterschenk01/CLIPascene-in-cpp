#ifndef PIXELFILTER_H_
#define PIXELFILTER_H_

#include <iostream>
#include <torch/torch.h>
#include "diffvg.h"
#include "filter.h"

struct PixelFilter {
    // Constructor
    PixelFilter(FilterType type, torch::Tensor radius = torch::tensor(0.5)) 
        : type(type), radius(radius) {}

    // Member variables
    FilterType type;
    torch::Tensor radius;
};

#endif