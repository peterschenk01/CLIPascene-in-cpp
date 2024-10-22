cmake_minimum_required(VERSION 3.2)

project("CLIPascene" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIBTORCH_DIR "/usr/local/libtorch")

set(CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(strokes_initialization strokes_initialization.cpp)
add_executable(libtorch_test libtorch_test.cpp)

target_link_libraries(strokes_initialization "${TORCH_LIBRARIES}")
target_link_libraries(libtorch_test "${TORCH_LIBRARIES}")