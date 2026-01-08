# CLIPascene-in-cpp

A C++ port of [**CLIPascene / SceneSketch**](https://github.com/yael-vinker/SceneSketch) with the long-term goal of running on Android.

This project was created as part of my bachelor's thesis.

## Repository layout

- `clipascene.cpp` — main executable entry point
- `CMakeLists.txt` — build configuration
- `download_model.py` — helper fetching the CLIP model
- `CLIP/` - CLIP implementation
- `diffvg/` - diffvg implementation
- `libdiffvg.so` — shared library of diffvg
- `tinyxml2/` - tinyxml implementation
- `helper/` — My C++ implementation of CLIPascene
- `input_images/` — inputs
- `output/` — outputs
- `tests/` — tests

## Requirements (Ubuntu / Debian)

> **Hint:** This implementation currently requires CUDA.

### 1. Install Libtorch

- Go to the official [PyTorch download page](https://pytorch.org/get-started/locally/) and select:

    - **Package**: LibTorch  
    - **Language**: C++  
    - **Compute platform**: CUDA (check which version you have installed)

- Download the archive (e.g. `libtorch-shared-with-deps-2.9.1%2Bcu128.zip`) into this directory

``` bash
curl -O https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.1%2Bcu128.zip
```

- Extract the archive

```bash
unzip libtorch-shared-with-deps-2.9.1%2Bcu128.zip
```

- Remove archive

``` bash
rm libtorch-shared-with-deps-2.9.1%2Bcu128.zip
```

### 2. Install OpenCV

```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev
```

### 3. Install CMake

``` bash
sudo apt-get update
sudo apt-get install -y cmake
```

### 4. Create python environment and install requirements

``` bash
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

## Running CLIPascene (C++)

### 1. Download CLIP model

```bash
python download_model.py
``` 

### 2. Build

``` bash
mkdir build
cd build
cmake ..
make
```

### 3. Run

``` bash
./clipascene
```

This will create following files in the `output/` directory:

- `attention_map.png`
- `distribution_map.png`
- `edge_map.png`
- `strokes_init.png`
- `strokes_init.svg`
