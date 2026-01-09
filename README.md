# CLIPascene-in-cpp

A C++ port of [**CLIPascene / SceneSketch**](https://github.com/yael-vinker/SceneSketch) with the long-term goal of running on Android.

This project was created as part of my bachelor's thesis.

---

> **! This project was not finished.** Below is a demonstration of what my implementation can do. !

---

### Input
<p align="center">
  <img src="https://github.com/user-attachments/assets/c6c7c47e-1559-4b54-a994-de6eea9ca504" width="320" />
</p>
<p align="center"><em>Input Image</em></p>

---

### Intermediate Maps
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3f4dc2fd-3460-46f6-88b3-d5880ee91ae5" width="224" /><br/>
      <strong>Attention Map</strong> <br/> <em>(CLIP)</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/4373a97f-6e3a-49b5-934d-17656850c36d" width="224" /><br/>
      <strong>Edge Map</strong> <br/> <em>(XDoG)</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/1f8a9488-f395-44d8-89d0-7314a4cdfb72" width="224" /><br/>
      <strong>Distribution Map</strong><br/>
      <em>(Attention + Edge)</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8c4ed4e3-8e1b-4a41-9932-2721f9f8a454" width="224" /><br/>
      <strong>Strokes Initialization</strong> <br/> <em>(rasterized with diffvg)</em>
    </td>
  </tr>
</table>

---

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

## Installation (Ubuntu / Debian)

> **Hint:** This implementation currently requires CUDA.

### 1. Clone this repository

``` bash
git clone https://github.com/peterschenk01/CLIPascene-in-cpp.git
cd CLIPascene-in-cpp
```

### 2. Install Libtorch

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

### 3. Install OpenCV

```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev
```

### 4. Install CMake

``` bash
sudo apt-get update
sudo apt-get install -y cmake
```

### 5. Create python environment and install requirements

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
