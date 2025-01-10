# Image Resizing Library
Aiming to develop an efficient resize function that supports nearest neighbor interpolation, both upscaling and downscaling operations, and improves runtime performance through multithreaded optimization. Additionally, it will provide more features and optimization strategies.

---

This repository realizes the image resizing function based on OpenCV. We develop the library with the following goals:

1. Support different interpolation methods, including **Nearest Neighbor Interpolation** and **Bilinear Interpolation**. Any ratio of the width and height is supported, whatever **amplification** or **shrinking** is supported.
2. Support different data type for both **Nearest Neighbor Interpolation** and **Bilinear Interpolation**, including **8UC1**, **8UC3**, **16UC1**, **16UC3**, **32FC1**, **32FC3**.
3. Support multithreading for both **Nearest Neighbor Interpolation** and **Bilinear Interpolation**. Use OpenCV's `cv::parallel_for_` to implement the multithreading.
4. SIMD optimization for **Nearest Neighbor Interpolation** Only. Use AVX2 instructions 256 bits registers based on the multithreading infrastructure provided by OpenCV, `cv::parallel_for_`. You will define the **`USE_AVX2`** automatically if your device support AVX2 instructions.
5. Reach **`TEST` mode** by adding **`-DTEST`** in the cmake command. And we use macros to manipulate the different mode and hardware acceleration. 

The design of Codebase is based on the [OpenCV's design](https://github.com/opencv/opencv/blob/4866811933ac9d188fe098308fb34112de296992/modules/imgproc/src/resize.cpp#L1121-L1172), and the SIMD optimization is based on the Intel's [AVX2 instructions](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avxnewtechs=AVX2).


### Dependencies

Before building or running this project, ensure the following requirements are met:

- **CMake >= 3.10:**
Required for configuring and generating build files.

- **C++ 11:**
Ensure your compiler supports at least the C++11 standard.

- **OpenCV >= 4.6.0:**
Used for image processing tasks; this exact version is required for compatibility. Follow the instructions below to install OpenCV.

  ```bash
  sudo apt-get install libopencv-dev
  ```

### Quick Start

```bash
git clone https://github.com/lizhuo008/resize_func.git
cd resize_func
mkdir build
cd build
cmake ..
make
./resize_func
```

### Project Structure

1. **Core Features**
   - `imgproc.hpp`: Main declarations for image resizing, implemented by `imgproc.cpp` and `simd.cpp`.
   - `imgproc.cpp`: Implementation of image resizing features.

2. **SIMD Optimizations**
   - `simd.cpp`: Specialized functions leveraging SIMD for performance.

3. **Utilities**
   - `utils.hpp`: Complementary utilities for supporting operations.
   - `utils.cpp`: Implementation of utilities.

4. **Testing**
   - `test.hpp`: Declarations for testing the library.
   - `test.cpp`: Test cases and validation for implemented functions.