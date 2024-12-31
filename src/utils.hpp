#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>

void createTestImage(cv::Mat& img, const cv::Size& size, int type = CV_8UC3, int seed = 0);

#endif