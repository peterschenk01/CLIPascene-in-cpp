#include <iostream>
#include "tensor.h"

using namespace std;

torch::Tensor randomTensor(){
    return torch::rand({2, 3});
}