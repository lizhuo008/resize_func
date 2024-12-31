#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>

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