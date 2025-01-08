#ifndef UTILS_HPP
#define UTILS_HPP
#include <opencv2/opencv.hpp>
#include <chrono>

#define TIME_START auto start = std::chrono::high_resolution_clock::now();
#define TIME_END(NAME) \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration<double, std::milli>(end - start).count(); \
    // std::cout << NAME << " time: " << duration << "ms" << std::endl;

void createTestImage(cv::Mat& img, const cv::Size& size, int type = CV_8UC3, int seed = 0);

#endif