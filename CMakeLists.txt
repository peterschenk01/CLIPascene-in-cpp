cmake_minimum_required(VERSION 3.12)

project("CLIPascene" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Libtorch
set(LIBTORCH_DIR "/usr/local/libtorch")
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# OpenCV
find_package(OpenCV REQUIRED)

include_directories(diffvg)

add_executable(clipascene 
                "clipascene.cpp"
                "helper/strokes_initialization.cpp"
                "helper/strokes_initialization.h"
                "helper/diffvg/cppdiffvg.cpp"
                "helper/diffvg/cppdiffvg.h"
                "helper/diffvg/pixel_filter.h"
                "helper/diffvg/path.h"
                )

target_link_libraries(clipascene PRIVATE "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")