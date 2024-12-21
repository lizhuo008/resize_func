#include <iostream>

#include <opencv2/opencv.hpp>
#include <imgproc.hpp>


void imgproc::resize_custom(cv::Mat &img, cv::Mat &resized_img, int new_width, int new_height)
{
    cv::resize(img, resized_img, cv::Size(new_width, new_height));
}