#ifndef UTILS_HPP
#define UTILS_HPP
#include <opencv2/opencv.hpp>
#include <chrono>

#define TIME_START auto start = std::chrono::high_resolution_clock::now();
#define TIME_END(NAME) \
    auto end = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration<double, std::milli>(end - start).count(); \
    // std::cout << NAME << " time: " << duration << "ms" << std::endl;

#define CVT_3C21C(img) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY)
#define CVT_1C23C(img) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR)
#define CVT_8U216U(img) img.convertTo(img, CV_16U, 256.0)
#define CVT_16U28U(img) img.convertTo(img, CV_8U, 1.0/256.0)
#define CVT_8U232F(img) img.convertTo(img, CV_32F, 1.0/255.0)
#define CVT_32F28U(img) img.convertTo(img, CV_8U, 255.0)

void createTestImage(cv::Mat& img, const cv::Size& size, int type = CV_8UC3, int seed = 0);

#endif