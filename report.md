# Image Resizing Library Documentation

**Author:** Lizhuo Luo, Jiachen Zhou

**Email:** <12111925@mail.sustech.edu.cn>

**Date:** 2025-01-09

---
## Introduction
This document provides a comprehensive overview of the components of the image resizing library. 

This project realizes the image resizing function based on OpenCV. We develop the library with the following goals:

1. Support different interpolation methods, including **Nearest Neighbor Interpolation** and **Bilinear Interpolation**. Any ratio of the width and height is supported, whatever **amplification** or **shrinking** is supported.
2. Support different data type for both **Nearest Neighbor Interpolation** and **Bilinear Interpolation**, including **8UC1**, **8UC3**, **16UC1**, **16UC3**, **32FC1**, **32FC3**.
3. Support multithreading for both **Nearest Neighbor Interpolation** and **Bilinear Interpolation**. Use OpenCV's `cv::parallel_for_` to implement the multithreading.
4. SIMD optimization for **Nearest Neighbor Interpolation** Only. Use AVX2 instructions 256 bits registers based on the multithreading infrastructure provided by OpenCV, `cv::parallel_for_`. 
5. Reach `TEST` mode by adding `-DTEST` in the cmake command. And we use macros to manipulate the different mode and hardware acceleration.

We open-source the project on github: https://github.com/lizhuo008/resize_func.git


**Dependencies**:

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

**Quick Start**:

```bash
git clone https://github.com/lizhuo008/resize_func.git
cd resize_func
mkdir build
cd build
cmake ..
make
./resize_func
```

## Project Structure

1. **Core Features**
   - `imgproc.hpp`: Main declarations for image resizing.
   - `imgproc.cpp`: Implementation of image resizing features.

2. **SIMD Optimizations**
   - `simd.cpp`: Specialized functions leveraging SIMD for performance.

3. **Utilities**
   - `utils.hpp`: Complementary utilities for supporting operations.
   - `utils.cpp`: Implementation of utilities.

4. **Testing**
   - `test.hpp`: Declarations for testing the library.
   - `test.cpp`: Test cases and validation for implemented functions.

---

## File: `imgproc.hpp`

### Class: `resizeNNInvoker_custom`

This class performs parallelized nearest-neighbor interpolation using OpenCV's `cv::parallel_for_`.

```cpp
template <typename T>
class resizeNNInvoker_custom : public cv::ParallelLoopBody
{
    public:
    resizeNNInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* x_ofs, double _ify);

    void operator()(const cv::Range& range) const CV_OVERRIDE;

    private:
    const cv::Mat& input;
    cv::Mat& output;
    const cv::Size& inp_size;
    const cv::Size& out_size;
    int* x_ofs;
    double ify;
};
```

- **Purpose**:  
  Enables parallel nearest-neighbor interpolation.
- **Key Method**:  
  `operator()(const cv::Range& range)` processes a specific range of rows for parallel computation.

---

### Additional Functions in `imgproc.hpp`

#### `resizeNN_custom`

```cpp
void resizeNN_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);
```

- **Purpose**:  
  Resizes an image using nearest-neighbor interpolation.

#### `resizeNN_naive`

```cpp
void resizeNN_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);
```

- **Purpose**:  
  A simple implementation of the Nearest Neighbor interpolation for image resizing, processed sequentially without parallelism.

#### `resizeBilinearInvoker_custom`

```cpp
class resizeBilinearInvoker_custom : public cv::ParallelLoopBody
```

- **Purpose**:  
  This class facilitates parallelized bilinear interpolation for image resizing. It divides the computation across multiple threads using OpenCV's `cv::parallel_for_`. The bilinear interpolation calculates the weighted average of the four nearest neighbors in the input image to determine the pixel value in the output image.


#### `resizeBilinear_custom`

```cpp
void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);
```

- **Purpose**:  
  Resizes an image using bilinear interpolation.

#### `resizeBilinear_naive`

```cpp
void resizeBilinear_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);
```

- **Purpose**:  
  This function implements a naive bilinear interpolation for resizing images. It calculates the weighted average of the four nearest neighbors for each output pixel, mapping them from the input image based on scaling factors. The function processes rows and columns sequentially without parallelization.


#### `resize_custom`

```cpp
void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation);
```

- **Purpose**:  
  A generic image resizing function supporting different interpolation methods.

---

## File: `imgproc.cpp`

This file contains the implementation of the core functions declared in `imgproc.hpp`.

---

### Implementation: `resizeNNInvoker_custom`

#### Constructor

```cpp
template <typename T>
resizeNNInvoker_custom<T>::resizeNNInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x_ofs, double _ify)
: input(_input), output(_output), inp_size(_inp_size), out_size(_out_size), x_ofs(_x_ofs), ify(_ify)
{
}
```

- **Purpose**: Initializes the object with the necessary input and output parameters.

#### Operator Overload

```cpp
template <typename T>
void resizeNNInvoker_custom<T>::operator()(const cv::Range& range) const
{
    // Implementation of the nearest-neighbor interpolation.
}
```

---

### Implementation: `resizeNN_custom`

```cpp
void resizeNN_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{
}
```

- **Purpose**: Customed resize function with Nearest Neighbor Interpolation using parallel computing.

---

### Implementation: `resizeBilinearInvoker_custom`

```cpp
void resizeBilinearInvoker_custom<T>::operator()(const cv::Range& range) const

```

- **Purpose**: This function implements the core logic for bilinear interpolation as part of the parallelized processing framework. It calculates the weighted average of four neighboring pixels for a specific range of rows in the output image, allowing efficient multi-threaded resizing.

---

### Implementation: `resizeBilinear_custom`

```cpp
void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
```

- **Purpose**: This function performs bilinear interpolation for resizing images using parallel processing. It determines the interpolation weights and offsets for all columns and delegates the row-wise computation to the `resizeBilinearInvoker_custom` class, enabling efficient multi-threaded execution.

---

### Implementation: `resize_naive`

```cpp
void resize_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation)
```

- **Purpose**: This function serves as a general interface for image resizing using naive interpolation methods. It supports both nearest-neighbor and bilinear interpolation

---

### Implementation: `resize_custom`

```cpp
void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation){}

```

- **Purpose**: The final customed resize function, intergrating Nearest Neighbor Interpolation and Bilinear Interpolation.

---

## File: `simd.cpp`

This file provides SIMD-optimized functions for image processing tasks. Functions here are in the `simd` namespace, declared in `imgproc.hpp`.

To reach the best performance, we use AVX2 instructions 256 bits registers based on the multithreading infrastructure provided by OpenCV, `cv::parallel_for_`.

---

### Class: `resizeNNInvoker_AVX2` 

This class is the SIMD version of the `resizeNNInvoker_custom` class, which is provide SIMD optimization using AVX2 instructions for the nearest-neighbor interpolation. 

Different data types are supported, and the class is templated. But the realization among different data types are huge different limited by the restriction of the register size and instruction set.

#### Constructor

```cpp
template <typename T>
simd::resizeNNInvoker_AVX2<T>::resizeNNInvoker_AVX2(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x_ofs, double _ify)
: input(_input), output(_output), inp_size(_inp_size), out_size(_out_size), x_ofs(_x_ofs), ify(_ify)
{
}
```

- **Purpose**: Initializes the object with the necessary input and output parameters for AVX2 optimization.

#### Operator Overload

The realization of the operator overload is the most important part of the class. The basic idea is to deal the image by pixel, and use the ``_mm256`` or ``_mm256`` instructions to **(1) get the x offset of the pixel, (2) gather the pixel data from the input image, (3) shuffle and permute the pixel data to get certain byte data. (4) store the data to the output image.** 

**WARNING:** The process of the pixel data is easy to write data out of the boundary of the image, so we need to be careful about the boundary check.

```cpp
template <typename T>
void simd::resizeNNInvoker_AVX2<T>::operator()(const cv::Range& range) const
{
    // Optimized nearest-neighbor interpolation using AVX2.
}
```

---

### Function: `resizeNN_AVX2` and `resize_AVX2` (Only for Experiment Purpose)

```cpp
void simd::resizeNN_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{
    // Implements SIMD-optimized resizing using AVX2.
}

void simd::resize_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation)
{
    // Implements SIMD-optimized resizing using AVX2.
}
```

- **Purpose**: Precompute the x offset for `resizeNNInvoker_AVX2` and this function has already been integrated into the `resize_custom` function.

---

## File: `utils.cpp`

This file provides utility functions to support the library's main features.

---

### Function: `createTestImage`

```cpp
void createTestImage(cv::Mat& img, const cv::Size& size, int type = CV_8UC3, int seed = 0)
{
    cv::RNG rng(seed);
    img = cv::Mat(size, type);

    assert(type == CV_8UC1 || type == CV_8UC3 || type == CV_16UC1 || type == CV_16UC3 || type == CV_32FC1 || type == CV_32FC3);

    switch (type)
    {
        case CV_8UC1:
        case CV_8UC3:
            rng.fill(img, cv::RNG::UNIFORM, 0, 255);
            break;
        case CV_16UC1:
        case CV_16UC3:
            rng.fill(img, cv::RNG::UNIFORM, 0, 65535);
            break;
        case CV_32FC1:
        case CV_32FC3:
            rng.fill(img, cv::RNG::UNIFORM, 0.0f, 1.0f);
            break;
    }
}
```

- **Purpose**: Generates a test image filled with random data based on the specified type and size.

- **Parameters**:
  - `img`: Reference to the output matrix.
  - `size`: Size of the generated image.
  - `type`: Data type of the image (default: `CV_8UC3`).
  - `seed`: Seed for random number generation (default: `0`).

- **Key Steps**:
  1. Ensures the image type is valid using `assert`.
  2. Fills the image with random values within a type-specific range using OpenCV's `RNG`.

---

## File: `utils.hpp`

This file declares utility functions and macros to support image processing and performance measurement.

---


### Macros

#### Timing Macros

```cpp
#define TIME_START auto start = std::chrono::high_resolution_clock::now();
#define TIME_END(NAME) \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration<double, std::milli>(end - start).count(); \
```

- **Purpose**: Provides an easy-to-use mechanism for measuring function execution time.

#### Image Conversion Macros

```cpp
#define CVT_3C21C(img) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY)
#define CVT_1C23C(img) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR)
#define CVT_8U216U(img) img.convertTo(img, CV_16U, 256.0)
#define CVT_16U28U(img) img.convertTo(img, CV_8U, 1.0/256.0)
#define CVT_8U232F(img) img.convertTo(img, CV_32F, 1.0/255.0)
#define CVT_32F28U(img) img.convertTo(img, CV_8U, 255.0)
```

- **Purpose**: Provide a easy way to convert the image type, which is useful for the test.

## File: `test.hpp` and `test.cpp`

### Function Declarations

This file includes tests for functionality based on the requirement.

---

#### `basic_test`

```cpp
void basic_test();
```

- **Purpose**: Tests basic resizing functionality with a single example image.

#### `multi_type_test`

```cpp
void multi_type_test(int interpolation = cv::INTER_NEAREST);
```

- **Purpose**: Validates the resizing functionality across various data types and channels using the Mat structure.

#### `amp_shr_test`

```cpp
void amp_shr_test(int interpolation = cv::INTER_NEAREST);
```

- **Purpose**: Tests amplification and shrinking of images to user-specified dimensions.

#### `multithread_test`

```cpp
void multithread_test(int interpolation = cv::INTER_NEAREST);
```

- **Purpose**: Compares single-threaded and parallel computing resizing performance.

#### `simd_test`

```cpp
void simd_test();
```

- **Purpose**: Benchmarks SIMD-based resizing against other implementations for performance evaluation.

#### `standard_comp_test`

```cpp
void standard_comp_test(int interpolation = cv::INTER_NEAREST);
```

- **Purpose**: Benchmarks custom resizing against OpenCV's native resizing functions.

---

### Test Result and Analysis

- **Experiment Environment Information**:
  - **CPU Model**: Intel(R) Core(TM) Ultra 5 125H
  - **Base Frequency**: 1.20 GHz
  - **Max Frequency**: 3.61 GHz
  - **Cores/Threads**: 14 cores / 18 threads
  - **L1 Cache**: 1.4 MB
  - **L2 Cache**: 14.0 MB
  - **L3 Cache**: 18.0 MB
  - **Hyper-Threading**: Enabled
  - **Virtualization**: Enabled
  - **CPU Utilization**: 100% during the experiment


#### `multithread_test`

- **Result**:

![Multithread Test Result using Nearest Neighbor Interpolation](test_result_sample/Figure_1_1.png)
![Multithread Test Result using Bilinear Interpolation](test_result_sample/Figure_1_2.png)

- **Analysis**: Parallelization has a significant impact on improving resize performance, especially for larger and higher precision images. The benefits of multithreading become more evident when dealing with formats such as 16UC1, 16UC3, 32FC1, and 32FC3, where the Naive Resize method performs considerably slower. While bilinear interpolation inherently requires more time than nearest neighbor, the parallel approach substantially mitigates this.

---

#### `simd_test`

- **Result**:

![SIMD Test Result](test_result_sample/Figure_2.png)

- **Analysis**: SIMD provides a clear advantage over custom resizing in all tested cases, demonstrating its efficiency, especially for larger image formats and multi-channel images.

---

#### `standard_comp_test`

- **Result for Nearest Neighbor Interpolation**:

![Standard Test Result for data type 0 using Nearest Neighbor Interpolation](test_result_sample/Figure_3_d0NN.png)
![Standard Test Result for data type 16 using Nearest Neighbor Interpolation](test_result_sample/Figure_3_d16NN.png)
![Standard Test Result for data type 18 using Nearest Neighbor Interpolation](test_result_sample/Figure_3_d18NN.png)

- **Analysis**: OpenCV is significantly more efficient than the Custom Resize method for nearest neighbor interpolation, showing clear advantages in terms of both speed and scalability as image size increases. The performance gain with OpenCV is particularly notable for larger images, with up to 55.9% faster resizing, making OpenCV a highly suitable choice for real-time applications or large-scale image processing tasks.

- **Result for Bilinear Interpolation**:

![Standard Test Result for data type 0 using Bilinear Interpolation](test_result_sample/Figure_3_d0BL.png)
![Standard Test Result for data type 16 using Bilinear Interpolation](test_result_sample/Figure_3_d16BL.png)
![Standard Test Result for data type 18 using Bilinear Interpolation](test_result_sample/Figure_3_d18BL.png)

- **Analysis**: OpenCV remains the faster and more efficient choice for bilinear interpolation across all test cases, with differences consistently ranging from 26.7% to 34.7% faster. The image size and data type both influence the magnitude of the performance difference, but OpenCV's optimizations allow it to consistently outperform Custom Resize, particularly for larger images and higher precision formats.

---