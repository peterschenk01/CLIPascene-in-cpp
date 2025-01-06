#include <iostream>
#include <chrono>
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
                                      torch::Tensor input,
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
    vector<ptr<Path>> path_pointers;
    vector<Path> path_contents;
    vector<ptr<Constant>> color_pointers;
    vector<Constant> color_contents;

    // Vectors to store the individual variables of the shapes and shape_groups structs because torch::AutogradContext can only store 
    // standard C++ types
    
    // Shapes
    vector<torch::Tensor> num_control_points_contents;
    vector<torch::Tensor> points_contents;
    vector<int> is_closed_contents;
    vector<int> use_distance_approx_contents;
    vector<int> shape_type_contents;
    vector<torch::Tensor> stroke_width_contents;
    // ShapeGroups
    vector<torch::Tensor> shape_ids_contents;
    vector<int> stroke_color_type_contents;
    vector<torch::Tensor> stroke_color_contents;
    vector<int> use_even_odd_rule_contents;
    vector<torch::Tensor> shape_to_canvas_contents;

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

        ptr<Path> path = new Path(num_control_points.data_ptr<int>(),
                                  points.data_ptr<float>(),
                                  nullptr, // because thickness is false
                                  static_cast<int>(num_control_points.size(0)),
                                  static_cast<int>(points.size(0)),
                                  is_closed,
                                  use_distance_approx
                                  );

        torch::Tensor stroke_width = get<torch::Tensor>(args.at(current_index));
        current_index += 1;

        const Shape* shape = new Shape(shape_type, 
                                       path->get_ptr(),
                                       stroke_width.item<float>());

        path_contents.push_back(*path.get());
        path_pointers.push_back(path);
        shapes.push_back(shape);

        num_control_points_contents.push_back(num_control_points);
        points_contents.push_back(points);
        is_closed_contents.push_back(is_closed);
        use_distance_approx_contents.push_back(use_distance_approx);
        shape_type_contents.push_back(static_cast<int>(shape_type));
        stroke_width_contents.push_back(stroke_width);
    }

    for(int i = 0; i < num_shape_groups; i++) {
        torch::Tensor shape_ids = get<torch::Tensor>(args.at(current_index));
        current_index += 1;
        bool fill_color_type = get<bool>(args.at(current_index)); // bool (false) because I won't use fill color
        current_index += 1;
        bool fill_color = false;
        ColorType stroke_color_type = get<ColorType>(args.at(current_index));
        current_index += 1;
        torch::Tensor color = get<torch::Tensor>(args.at(current_index));
        ptr<Constant> stroke_color = new Constant(Vector4f(color.index({0}).item<float>(), 
                                                           color.index({1}).item<float>(), 
                                                           color.index({2}).item<float>(), 
                                                           color.index({3}).item<float>())
                                                           );
        current_index += 1;
        bool use_even_odd_rule = get<bool>(args.at(current_index));
        current_index += 1;
        torch::Tensor shape_to_canvas = get<torch::Tensor>(args.at(current_index));
        current_index += 1;

        const ShapeGroup* shape_group = new ShapeGroup(shape_ids.data_ptr<int>(),
                                                       static_cast<int>(shape_ids.size(0)),
                                                       ColorType::Constant,
                                                       nullptr,
                                                       stroke_color_type,
                                                       stroke_color->get_ptr(),
                                                       use_even_odd_rule,
                                                       shape_to_canvas.data_ptr<float>()
                                                       );

        color_contents.push_back(*stroke_color.get());
        color_pointers.push_back(stroke_color);
        shape_groups.push_back(shape_group);

        shape_ids_contents.push_back(shape_ids);
        stroke_color_type_contents.push_back(static_cast<int>(stroke_color_type));
        stroke_color_contents.push_back(color);
        use_even_odd_rule_contents.push_back(use_even_odd_rule);
        shape_to_canvas_contents.push_back(shape_to_canvas);
    }

    FilterType filter_type = get<FilterType>(args.at(current_index));
    current_index += 1;
    torch::Tensor filter_radius = get<torch::Tensor>(args.at(current_index));
    current_index += 1;
    Filter filt = Filter(filter_type, filter_radius.item<float>());

    std::shared_ptr<Scene> scene = std::make_shared<Scene>(canvas_width,
                                                           canvas_height,
                                                           shapes, 
                                                           shape_groups,
                                                           filt, 
                                                           true,
                                                           static_cast<int>(torch::Device("cuda").index())
                                                           );
                    
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

    const char* diffvg_library_path = "libdiffvg.so";
    void* handle = dlopen(diffvg_library_path, RTLD_LAZY);
    if(!handle) {
        std::cerr << "Error: Unable to load library: " << dlerror() << std::endl;
    }

    dlerror();

    typedef void (*void_func_t)(std::shared_ptr<Scene>,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                int,
                                int,
                                int,
                                int,
                                unsigned long,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                bool,
                                ptr<float>,
                                int
                                );

    void_func_t render = (void_func_t)dlsym(handle, /*render function in libdiffvg.so*/ "_Z6renderSt10shared_ptrI5SceneE3ptrIfES3_S3_iiiimS3_S3_S3_S3_bS3_i"); 
    const char* error = dlerror();
    if (error != nullptr) {
        std::cerr << "Error: Unable to find function: " << error << std::endl;
        dlclose(handle);
    }

    auto start = std::chrono::high_resolution_clock::now();

    render(scene,
           nullptr,
           (output_type == OutputType::color) ? rendered_image.data_ptr<float>() : nullptr,
           (output_type == OutputType::sdf) ? rendered_image.data_ptr<float>() : nullptr,
           width,
           height,
           num_samples_x,
           num_samples_y,
           seed,
           nullptr, // d_background_image
           nullptr, // d_render_image
           nullptr, // d_render_sdf
           nullptr, // d_translation
           use_prefiltering,
           eval_positions.data_ptr<float>(),
           eval_positions.size(0)
           );

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Diffvg forward pass, time: " << duration.count() << " microseconds" << std::endl;

    dlclose(handle);

    assert(torch::isfinite(rendered_image).all().item<bool>());

    for (auto color : color_pointers) {
        color.destroy();
    }
    for (auto path : path_pointers) {
        path.destroy();
    }
    for (auto shape : shapes) {
        delete shape;
    }
    for (auto shape_group : shape_groups) {
        delete shape_group;
    }
    
    ctx->saved_data["canvas_width"] = canvas_width;
    ctx->saved_data["canvas_height"] = canvas_height;

    // Shapes
    ctx->saved_data["num_shapes"] = num_shapes;
    ctx->saved_data["num_control_points_contents"] = num_control_points_contents;
    ctx->saved_data["points_contents"] = points_contents;
    ctx->saved_data["is_closed_contents"] = torch::tensor(is_closed_contents, torch::kInt32);
    ctx->saved_data["use_distance_approx_contents"] = torch::tensor(use_distance_approx_contents, torch::kInt32);
    ctx->saved_data["shape_type_contents"] = torch::tensor(shape_type_contents, torch::kInt32);
    ctx->saved_data["stroke_width_contents"] = stroke_width_contents;

    // ShapeGroups
    ctx->saved_data["num_shape_groups"] = num_shape_groups;
    ctx->saved_data["shape_ids_contents"] = shape_ids_contents;
    ctx->saved_data["stroke_color_type_contents"] = torch::tensor(stroke_color_type_contents, torch::kInt32);
    ctx->saved_data["stroke_color_contents"] = stroke_color_contents;
    ctx->saved_data["use_even_odd_rule_contents"] = torch::tensor(use_even_odd_rule_contents, torch::kInt32);
    ctx->saved_data["shape_to_canvas_contents"] = shape_to_canvas_contents;

    // Filter
    ctx->saved_data["filter_type_int"] = static_cast<int>(filter_type);
    ctx->saved_data["filter_radius"] = filter_radius;

    ctx->saved_data["width"] = width;
    ctx->saved_data["height"] = height;
    ctx->saved_data["num_samples_x"] = num_samples_x;
    ctx->saved_data["num_samples_y"] = num_samples_y;
    ctx->saved_data["seed"] = seed;
    ctx->saved_data["output_type_int"] = static_cast<int>(output_type);
    ctx->saved_data["use_prefiltering"] = use_prefiltering;
    ctx->saved_data["eval_positions"] = eval_positions;
    ctx->saved_data["background_image"] = 0;

    return rendered_image * input;
}

torch::autograd::tensor_list RenderFunction::backward(torch::autograd::AutogradContext *ctx,
                                                      torch::autograd::tensor_list grad_img_list
                                                      ) {

    torch::Tensor grad_img = grad_img_list[0];

    if(!grad_img.is_contiguous()) {
        grad_img = grad_img.contiguous();
    }
    assert(torch::isfinite(grad_img).all().item<bool>());

    auto canvas_width = ctx->saved_data["canvas_width"].toInt();
    auto canvas_height = ctx->saved_data["canvas_height"].toInt();

    // Shapes
    auto num_shapes = ctx->saved_data["num_shapes"].toInt();
    auto num_control_points_contents = ctx->saved_data["num_control_points_contents"].toTensorVector();
    auto points_contents = ctx->saved_data["points_contents"].toTensorVector();
    auto is_closed_contents = ctx->saved_data["is_closed_contents"].toTensor();
    auto use_distance_approx_contents = ctx->saved_data["use_distance_approx_contents"].toTensor();
    auto shape_type_contents = ctx->saved_data["shape_type_contents"].toTensor();
    auto stroke_width_contents = ctx->saved_data["stroke_width_contents"].toTensorVector();

    // ShapeGroups
    auto num_shape_groups = ctx->saved_data["num_shape_groups"].toInt();
    auto shape_ids_contents = ctx->saved_data["shape_ids_contents"].toTensorVector();
    auto stroke_color_type_contents = ctx->saved_data["stroke_color_type_contents"].toTensor();
    auto stroke_color_contents = ctx->saved_data["stroke_color_contents"].toTensorVector();
    auto use_even_odd_rule_contents = ctx->saved_data["use_even_odd_rule_contents"].toTensor();
    auto shape_to_canvas_contents = ctx->saved_data["shape_to_canvas_contents"].toTensorVector();

    // Filter
    auto filter_type_int = ctx->saved_data["filter_type_int"].toInt();
    auto filter_radius = ctx->saved_data["filter_radius"].toTensor();

    auto width = ctx->saved_data["width"].toInt();
    auto height = ctx->saved_data["height"].toInt();
    auto num_samples_x = ctx->saved_data["num_samples_x"].toInt();
    auto num_samples_y = ctx->saved_data["num_samples_y"].toInt();
    auto seed = ctx->saved_data["seed"].toInt();
    auto output_type_int = ctx->saved_data["output_type_int"].toInt();
    auto use_prefiltering = ctx->saved_data["use_prefiltering"].toBool();
    auto eval_positions = ctx->saved_data["eval_positions"].toTensor();
    auto background_image = ctx->saved_data["background_image"];

    OutputType output_type;
    if(output_type_int == 0)
        output_type = OutputType::color;
    else if(output_type_int == 1)
        output_type = OutputType::sdf;
    else
        assert(false);

    vector<const Shape*> shapes;
    vector<const ShapeGroup*> shape_groups;
    vector<ptr<Path>> path_pointers;
    vector<ptr<Constant>> color_pointers;
        
    for(int i = 0; i < num_shapes; i++) {
        ptr<Path> path = new Path(num_control_points_contents[i].data_ptr<int>(),
                                  points_contents[i].data_ptr<float>(),
                                  nullptr, // because thickness is false
                                  static_cast<int>(num_control_points_contents[i].size(0)),
                                  static_cast<int>(points_contents[i].size(0)),
                                  is_closed_contents.index({i}).item<bool>(),
                                  use_distance_approx_contents.index({i}).item<bool>()
                                  );

        ShapeType shape_type;
        if(shape_type_contents.index({i}).item<int>() == 0)
            shape_type = ShapeType::Circle;
        else if(shape_type_contents.index({i}).item<int>() == 1)
            shape_type = ShapeType::Ellipse;
        else if(shape_type_contents.index({i}).item<int>() == 2)
            shape_type = ShapeType::Path;
        else if(shape_type_contents.index({i}).item<int>() == 3)
            shape_type = ShapeType::Rect;
        else
            assert(false);

        const Shape* shape = new Shape(shape_type, 
                                       path->get_ptr(),
                                       stroke_width_contents[i].item<float>());

        path_pointers.push_back(path);
        shapes.push_back(shape);
    }

    for(int i = 0; i < num_shape_groups; i++) {
        ColorType stroke_color_type;
        if(stroke_color_type_contents.index({i}).item<int>() == 0)
            stroke_color_type = ColorType::Constant;
        else if(stroke_color_type_contents.index({i}).item<int>() == 1)
            stroke_color_type = ColorType::LinearGradient;
        else if(stroke_color_type_contents.index({i}).item<int>() == 2)
            stroke_color_type = ColorType::RadialGradient;
        else
            assert(false);

        ptr<Constant> stroke_color = new Constant(Vector4f(stroke_color_contents[i].index({0}).item<float>(), 
                                                           stroke_color_contents[i].index({1}).item<float>(), 
                                                           stroke_color_contents[i].index({2}).item<float>(), 
                                                           stroke_color_contents[i].index({3}).item<float>())
                                                           );

        const ShapeGroup* shape_group = new ShapeGroup(shape_ids_contents[i].data_ptr<int>(),
                                                       static_cast<int>(shape_ids_contents[i].size(0)),
                                                       ColorType::Constant,
                                                       nullptr,
                                                       stroke_color_type,
                                                       stroke_color->get_ptr(),
                                                       use_even_odd_rule_contents.index({i}).item<bool>(),
                                                       shape_to_canvas_contents[i].data_ptr<float>()
                                                       );

        shape_groups.push_back(shape_group);
        color_pointers.push_back(stroke_color);
    }

    FilterType filter_type;
    if(filter_type_int == 0)
        filter_type = FilterType::Box;
    else if(filter_type_int == 1)
        filter_type = FilterType::Tent;
    else if(filter_type_int == 2)
        filter_type = FilterType::RadialParabolic;
    else if(filter_type_int == 3)
        filter_type = FilterType::Hann;
    else
        assert(false);
    
    Filter filt = Filter(filter_type, filter_radius.item<float>());

    std::shared_ptr<Scene> scene = std::make_shared<Scene>(canvas_width,
                                                           canvas_height,
                                                           shapes, 
                                                           shape_groups,
                                                           filt, 
                                                           true,
                                                           static_cast<int>(torch::Device("cuda").index())
                                                           );

    const char* diffvg_library_path = "libdiffvg.so";
    void* handle = dlopen(diffvg_library_path, RTLD_LAZY);
    if(!handle) {
        std::cerr << "Error: Unable to load library: " << dlerror() << std::endl;
    }

    dlerror();

    typedef void (*void_func_t)(std::shared_ptr<Scene>,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                int,
                                int,
                                int,
                                int,
                                unsigned long,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                ptr<float>,
                                bool,
                                ptr<float>,
                                int
                                );

    void_func_t render = (void_func_t)dlsym(handle, /*render function in libdiffvg.so*/ "_Z6renderSt10shared_ptrI5SceneE3ptrIfES3_S3_iiiimS3_S3_S3_S3_bS3_i"); 
    const char* error = dlerror();
    if (error != nullptr) {
        std::cerr << "Error: Unable to find function: " << error << std::endl;
        dlclose(handle);
    }

    auto start = std::chrono::high_resolution_clock::now();

    render(scene,
           nullptr, // background_image
           nullptr, // render_image
           nullptr, // render_sdf
           width,
           height,
           num_samples_x,
           num_samples_y,
           seed,
           nullptr, //d_background_image
           (output_type == OutputType::color) ? grad_img.data_ptr<float>() : nullptr,
           (output_type == OutputType::sdf) ? grad_img.data_ptr<float>() : nullptr,
           nullptr, // d_translation
           use_prefiltering,
           eval_positions.data_ptr<float>(),
           eval_positions.size(0)
           );

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Diffvg backward pass, time: " << duration.count() << " microseconds" << std::endl;

    dlclose(handle);

    for (auto color : color_pointers) {
        color.destroy();
    }
    for (auto path : path_pointers) {
        path.destroy();
    }
    for (auto shape : shapes) {
        delete shape;
    }
    for (auto shape_group : shape_groups) {
        delete shape_group;
    }

    torch::autograd::tensor_list d_args;

    d_args.push_back(torch::Tensor()); // width
    d_args.push_back(torch::Tensor()); // height
    d_args.push_back(torch::Tensor()); // num_samples_x
    d_args.push_back(torch::Tensor()); // num_samples_y
    d_args.push_back(torch::Tensor()); // seed
    d_args.push_back(torch::Tensor()); // d_background_image
    d_args.push_back(torch::Tensor()); // canvas_width
    d_args.push_back(torch::Tensor()); // canvas_height
    d_args.push_back(torch::Tensor()); // num_shapes
    d_args.push_back(torch::Tensor()); // num_shape_groups
    d_args.push_back(torch::Tensor()); // output_type
    d_args.push_back(torch::Tensor()); // use_prefiltering
    d_args.push_back(torch::Tensor()); // eval_positions

    for(int i = 0; i < scene->num_shapes; i++) {
        d_args.push_back(torch::Tensor()); // type
        Shape d_shape = scene->get_d_shape(i);
        bool use_thickness = false;

        // Normally you would differentiate between the possible shape types but we only use paths
        if(d_shape.type == ShapeType::Path) {
            Path d_path = d_shape.as_path();
            torch::Tensor points = torch::zeros((d_path.num_points, 2));
            torch::Tensor thickness = torch::Tensor();
            if(d_path.has_thickness()) {
                use_thickness = true;
                thickness = torch::zeros({d_path.num_points});
                d_path.copy_to(points.data_ptr<float>(), thickness.data_ptr<float>());
            } else {
                d_path.copy_to(points.data_ptr<float>(), nullptr);
            }
            assert(torch::isfinite(points).all().item<bool>());
            if(thickness.defined())
                assert(torch::isfinite(thickness).all().item<bool>());
            d_args.push_back(torch::Tensor()); // num_control_points
            d_args.push_back(points);
            d_args.push_back(thickness);
            d_args.push_back(torch::Tensor()); // is_closed
            d_args.push_back(torch::Tensor()); // use_distance_approx
        } else {
            assert(false);
        }

        if(use_thickness) {
            d_args.push_back(torch::Tensor());
        } else {
            torch::Tensor w = torch::tensor({d_shape.stroke_width});
            assert(torch::isfinite(w).all().item<bool>());
            d_args.push_back(w);
        }
    }

    for(int i = 0; i < scene->num_shape_groups; i++) {
        ShapeGroup d_shape_group = scene->get_d_shape_group(i);
        d_args.push_back(torch::Tensor()); // shape_ids
        d_args.push_back(torch::Tensor()); // fill_color_type

        if(d_shape_group.has_fill_color()) {
            // d_shape_group shouldn't have a fill color in our case
        }

        d_args.push_back(torch::Tensor()); // stroke_color_type

        if(d_shape_group.has_stroke_color()) {
            // Normally you would differentiate between the possible stroke color types but we only use ColorType::Constant
            if(d_shape_group.stroke_color_type == ColorType::Constant){
                auto d_constant = d_shape_group.stroke_color_as_constant();
                auto c = d_constant.color;
                d_args.push_back(torch::tensor({c.x, c.y, c.z, c.w}));
            } else {
                assert(false);
            }
        }

        d_args.push_back(torch::Tensor()); // use_even_odd_rule
        torch::Tensor d_shape_to_canvas = torch::zeros({3, 3});
        d_shape_group.copy_to(d_shape_to_canvas.data_ptr<float>());
        assert(torch::isfinite(d_shape_to_canvas).all().item<bool>());
        d_args.push_back(d_shape_to_canvas);
    }
    
    d_args.push_back(torch::Tensor()); // filter_type
    d_args.push_back(torch::tensor({scene->get_d_filter_radius()}));

    return d_args;
}