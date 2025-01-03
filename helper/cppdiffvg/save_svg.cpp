#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include "tinyxml2.h"
#include "path.h"

using namespace std;
using namespace tinyxml2;

void save_svg(const string filename,
              int width,
              int height,
              vector<PathCPP> paths
              ) {
    
    XMLDocument doc;

    XMLElement* root = doc.NewElement("svg");
    root->SetAttribute("version", "1.1");
    root->SetAttribute("xmlns", "http://www.w3.org/2000/svg");
    root->SetAttribute("width", width);
    root->SetAttribute("height", height);
    doc.InsertFirstChild(root);

    XMLElement* defs = doc.NewElement("defs");
    root->InsertEndChild(defs);
    XMLElement* g = doc.NewElement("g");
    root->InsertEndChild(g);

    for(auto path : paths) {
        XMLElement* path_node = doc.NewElement("path");

        int num_segments = path.num_control_points.size(0);
        torch::Tensor num_control_points = path.num_control_points;
        torch::Tensor points = path.points;
        int num_points = points.size(0);
        string path_str = "M "
                          + to_string(points.index({0, 0}).item<int>()) + " "
                          + to_string(points.index({0, 1}).item<int>());
        int point_id = 1;
        int p2 = (point_id + 2) % num_points;
        path_str += " C " 
                    + to_string(points.index({point_id, 0}).item<int>()) + " "
                    + to_string(points.index({point_id, 1}).item<int>()) + " "
                    + to_string(points.index({point_id + 1, 0}).item<int>()) + " "
                    + to_string(points.index({point_id + 1, 1}).item<int>()) + " "
                    + to_string(points.index({p2, 0}).item<int>()) + " "
                    + to_string(points.index({p2, 1}).item<int>());
        path_node->SetAttribute("d", path_str.c_str());
        path_node->SetAttribute("stroke_width", to_string(2 * path.stroke_width.item<int>()).c_str());
        path_node->SetAttribute("fill", "none");
        path_node->SetAttribute("stroke", "rgb(0, 0, 0)");
        path_node->SetAttribute("stroke-opacity", "1");
        path_node->SetAttribute("stroke-linecap", "round");
        path_node->SetAttribute("stroke-linejoin", "round");
        g->InsertEndChild(path_node);
    }

    XMLError result = doc.SaveFile(filename.c_str());

    if (result == XML_SUCCESS) {
        printf("SVG file saved successfully!\n");
    } else {
        printf("Error saving SVG file: %d\n", result);
    }
}

/*

    // Add shapes to the group
    for (size_t i = 0; i < shapeGroups.size(); ++i) {
        const auto& shape = shapes[shapeGroups[i]->shapeIds[0]];
        XMLElement* shapeNode = nullptr;
        } else if (auto path = dynamic_cast<Path*>(shape.get())) {
            shapeNode = doc.NewElement("path");
            shapeNode->SetAttribute("d", path->toSVGPath().c_str());
        }

        if (shapeNode) {
            g->InsertEndChild(shapeNode);
        }
    }

    // Save to file
    doc.SaveFile(filename.c_str());
}
*/