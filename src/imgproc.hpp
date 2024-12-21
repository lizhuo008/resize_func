#ifndef IMGPROC_HPP
#define IMGPROC_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

namespace imgproc
{
    void resize_custom(cv::Mat &img, cv::Mat &resized_img, int new_width, int new_height);
}

#endif