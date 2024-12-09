#include <iostream>
#include <typeinfo>
#include <cassert>
#include <dlfcn.h>
#include <torch/torch.h>
#include "cppdiffvg.h"
#include "diffvg.h"
#include "filter.h"
#include "pixel_filter.h"
#include "path.h"
#include "shape.h"
#include "scene.h"

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
                                                       vector<PathCPP> paths,
                                                       vector<PathGroupCPP> path_groups,
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

    for(PathCPP pth : paths) {
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

    for(PathGroupCPP grp : path_groups) {
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
                                      // torch::Tensor background_image,
                                      vector<custom_variant> args
                                      ) {
    
    // Unpack arguments
    int current_index = 0;
    int canvas_width = get<int>(args.at(current_index));
    current_index += 1;
    int canvas_height = get<int>(args.at(current_index));
    current_index += 1;
    int num_shapes = get<int>(args.at(current_index));
    current_index += 1;
    int num_shape_groups = get<int>(args.at(current_index));
    current_index += 1;
    OutputType output_type = get<OutputType>(args.at(current_index));
    current_index += 1;
    bool use_prefiltering = get<bool>(args.at(current_index));
    current_index += 1;
    torch::Tensor eval_positions = get<torch::Tensor>(args.at(current_index));
    current_index += 1;

    vector<const Shape*> shapes;
    vector<const ShapeGroup*> shape_groups;
    vector<Path> path_contents;
    vector<torch::Tensor> color_contents;

    for(int i = 0; i < num_shapes; i++) {
        ShapeType shape_type = get<ShapeType>(args.at(current_index));
        current_index += 1;
        torch::Tensor num_control_points = get<torch::Tensor>(args.at(current_index));
        current_index += 1;
        torch::Tensor points = get<torch::Tensor>(args.at(current_index));
        current_index += 1;
        bool thickness = get<bool>(args.at(current_index)); // bool because the Paths we are using never use thickness (always false in our case)
        current_index += 1;
        bool is_closed = get<bool>(args.at(current_index));
        current_index += 1;
        bool use_distance_approx = get<bool>(args.at(current_index));
        current_index += 1;
        Path path = Path(num_control_points.data_ptr<int>(),
                         points.data_ptr<float>(),
                         0.0, // because thickness is false
                         num_control_points.size(0),
                         points.size(0),
                         is_closed,
                         use_distance_approx
                         );
        torch::Tensor stroke_width = get<torch::Tensor>(args.at(current_index));
        current_index += 1;

        Shape* shape = new Shape(shape_type, 
                                 path.get_ptr(), 
                                 stroke_width.item<float>());
        shapes.push_back(shape);
        path_contents.push_back(path);
    }

    for(int i = 0; i < num_shape_groups; i++) {
        torch::Tensor shape_ids = get<torch::Tensor>(args.at(current_index));
        current_index += 1;
        bool fill_color_type = get<bool>(args.at(current_index)); // bool (false) because I won't use fill color
        current_index += 1;
        bool fill_color = false;
        ColorType stroke_color_type = get<ColorType>(args.at(current_index));
        current_index += 1;
        torch::Tensor stroke_color = get<torch::Tensor>(args.at(current_index));
        current_index += 1;
        bool use_even_odd_rule = get<bool>(args.at(current_index));
        current_index += 1;
        torch::Tensor shape_to_canvas = get<torch::Tensor>(args.at(current_index));
        current_index += 1;
        color_contents.push_back(stroke_color);

        ShapeGroup* shape_group = new ShapeGroup(shape_ids.data_ptr<int>(),
                                                 shape_ids.size(0),
                                                 ColorType::Constant,
                                                 ptr<void>(static_cast<size_t>(0)),
                                                 stroke_color_type,
                                                 stroke_color.data_ptr<float>(),
                                                 use_even_odd_rule,
                                                 shape_to_canvas.data_ptr<float>()
                                                 );
        shape_groups.push_back(shape_group);
    }

    FilterType filter_type = get<FilterType>(args.at(current_index));
    current_index += 1;
    torch::Tensor filter_radius = get<torch::Tensor>(args.at(current_index));
    current_index += 1;
    Filter filt = Filter(filter_type, filter_radius.item<float>());


    /*const char* diffvg_library_path = "../build/diffvg/diffvg.so";
    void* handle = dlopen(diffvg_library_path, RTLD_LAZY);
    if(!handle) {
        std::cerr << "Error: Unable to load library: " << dlerror() << std::endl;
    }

    // Clear any existing errors
    dlerror();

    typedef Scene* (*create_scene_func)(int, int, const std::vector<const Shape*>&, const std::vector<const ShapeGroup*>&, const Filter&, bool, int);
    create_scene_func create_scene = (create_scene_func) dlsym(handle, "create_scene");
    if (!create_scene) {
        std::cerr << "Error finding symbol: " << dlerror() << std::endl;
        dlclose(handle);
    }*/

    for(auto shape : shapes) {
        // cout << typeid(shape->type).name() << endl;
        if(shape->type == ShapeType::Path) {
            cout << "Path" << endl;
        }
    }

    for(auto shape_group : shape_groups) {
        // cout << typeid(shape->type).name() << endl;
        if(shape_group->fill_color_type == ColorType::Constant) {
            cout << "Constant" << endl;
        }
    }

    Scene scene = Scene(canvas_width,
                        canvas_height,
                        shapes, 
                        shape_groups,
                        filt, 
                        true,
                        static_cast<int>(torch::Device("cuda").index())
                        );

    // std::shared_ptr<Scene> scene_ptr = std::make_shared<Scene>(scene);

    /*Scene* scene = create_scene(canvas_width, 
                                                            canvas_height,
                                                            shapes, 
                                                            shape_groups,
                                                            filt, 
                                                            true,
                                                            static_cast<int>(torch::Device("cuda").index())
                                                            );

    std::shared_ptr<Scene> scene_ptr = std::make_shared<Scene>(*scene);*/

    /*std::shared_ptr<Scene> scene_ptr = std::shared_ptr<Scene>(create_scene(canvas_width, 
                                                            canvas_height,
                                                            shapes, 
                                                            shape_groups,
                                                            filt, 
                                                            true,
                                                            static_cast<int>(torch::Device("cuda").index()))
                                                            );*/
    // shared_ptr<Scene> scene = std::make_shared<Scene>                                                        

    torch::Tensor rendered_image;
    if(output_type == OutputType::color) {
        assert(eval_positions.size(0) == 0);
        rendered_image = torch::zeros({height, width, 4}, torch::Device("cuda"));
    } else {
        assert(output_type == OutputType::sdf);
        if(eval_positions.size(0) == 0) {
            rendered_image = torch::zeros({height, width, 1}, torch::Device("cuda"));
        } else {
            rendered_image = torch::zeros({eval_positions.size(0), 1}, torch::Device("cuda"));
        }
    }

    // No background image

    /*typedef void (*void_func_t)(std::shared_ptr<Scene>,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                int,
                                int,
                                int,
                                int,
                                uint64_t,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                bool,
                                ptr<float>,
                                int
                                );
    void_func_t render = (void_func_t)dlsym(handle, "render");
    const char* error = dlerror();
    if (error != nullptr) {
        std::cerr << "Error: Unable to find function: " << error << std::endl;
        dlclose(handle);
    }*/

    /*render(std::shared_ptr<Scene>(),
           nullptr,
           (output_type == OutputType::color) ? rendered_image.data_ptr<float>() : nullptr,
           (output_type == OutputType::sdf) ? rendered_image.data_ptr<float>() : nullptr,
           width,
           height,
           num_samples_x,
           num_samples_y,
           seed,
           ptr<float>(static_cast<std::size_t>(0)), // d_background_image
           ptr<float>(static_cast<std::size_t>(0)), // d_render_image
           ptr<float>(static_cast<std::size_t>(0)), // d_render_sdf
           ptr<float>(static_cast<std::size_t>(0)), // d_translation
           use_prefiltering,
           eval_positions.data_ptr<float>(),
           eval_positions.size(0)
           );*/
    
    assert(torch::isfinite(rendered_image).all().item<bool>());

    // dlclose(handle);

    /*ctx->save_for_backward({
        scene,
        // background_image,
        path_contents,
        color_contents,
        filt,
        width,
        height,
        num_samples_x,
        num_samples_y,
        seed,
        output_type,
        use_prefiltering,
        eval_positions
    });*/

    // ctx->saved_data["scene"] = scene;

    for (auto shape : shapes) {
        delete shape;
    }

    for (auto shape_group : shape_groups) {
        delete shape_group;
    }

    return rendered_image;
}

torch::autograd::tensor_list RenderFunction::backward(torch::autograd::AutogradContext *ctx,
                                                      torch::autograd::tensor_list grad_img
                                                      ) {

    return {};
}