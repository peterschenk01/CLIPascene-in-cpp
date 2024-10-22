cmake_minimum_required(VERSION 3.2)

project("CLIPascene" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBTORCH_DIR "/usr/local/libtorch")

set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(clipascene 
                clipascene.cpp
                helper/tensor.cpp
                helper/tensor.h
                )

target_link_libraries(clipascene "${TORCH_LIBRARIES}")