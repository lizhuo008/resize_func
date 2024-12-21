#include <iostream>
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>

int main(void)
{
    cv::Mat img = cv::imread("../samples/img.jpg");
    assert(!img.empty());

    imgproc::resize_custom(img, img, 100, 100);

    cv::imshow("Resized Image", img);
    cv::waitKey(0);
    return 0;
}