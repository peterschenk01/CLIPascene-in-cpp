cmake_minimum_required(VERSION 3.18)

project("CLIPascene" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Libtorch
set(LIBTORCH_DIR "/usr/local/libtorch")
set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# OpenCV
find_package(OpenCV REQUIRED)

# DiffVG
include_directories(diffvg)
set(CMAKE_BUILD_RPATH "${CMAKE_SOURCE_DIR}/libdiffvg.so")
set(diffvg_src
    diffvg/shape.cpp)

# CPPDiffvg
set(cppdiffvg_src
    helper/cppdiffvg/cppdiffvg.cpp
    helper/cppdiffvg/save_image.cpp
    helper/cppdiffvg/cppdiffvg.h
    helper/cppdiffvg/save_image.h
    helper/cppdiffvg/pixel_filter.h
    helper/cppdiffvg/path.h)

# TinyXML-2
include_directories(tinyxml2)
set(tinyxml2_src
    tinyxml2/tinyxml2.cpp
    tinyxml2/tinyxml2.h)

# CLIPascene
set(clipascene_src
    clipascene.cpp
    helper/strokes_initialization.cpp
    helper/strokes_initialization.h)

# Final Executable
add_executable(clipascene 
    ${clipascene_src}
    ${cppdiffvg_src}
    ${diffvg_src}
    ${tinyxml2_src}
)

target_link_libraries(clipascene 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS}
    ${CMAKE_SOURCE_DIR}/libdiffvg.so # Diffvg shared library
    /usr/lib/python3.10/config-3.10-x86_64-linux-gnu/libpython3.10.so # Python C API for Diffvg shared library
)