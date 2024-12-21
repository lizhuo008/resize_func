#ifndef IMGPROC_HPP
#define IMGPROC_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

namespace imgproc
{
    void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation = cv::INTER_NEAREST);
}

#endif