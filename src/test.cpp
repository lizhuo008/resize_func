#include <iostream> 
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>
#include <test.hpp>
#include <utils.hpp>

using namespace std;

void multi_type_test()
{
    cv::Mat testImg_8UC1, testImg_8UC3, testImg_16UC1, testImg_16UC3, testImg_32FC1, testImg_32FC3;
    cv::Mat testImg_8UC1_res, testImg_8UC3_res, testImg_16UC1_res, testImg_16UC3_res, testImg_32FC1_res, testImg_32FC3_res;
    cv::Size size(1000, 1000);

    createTestImage(testImg_8UC1, size, CV_8UC1);
    createTestImage(testImg_8UC3, size, CV_8UC3);
    createTestImage(testImg_16UC1, size, CV_16UC1);
    createTestImage(testImg_16UC3, size, CV_16UC3);
    createTestImage(testImg_32FC1, size, CV_32FC1);
    createTestImage(testImg_32FC3, size, CV_32FC3);

    cv::Size new_size(500, 500);

    resize_custom(testImg_8UC1, testImg_8UC1_res, new_size, cv::INTER_NEAREST);
    resize_custom(testImg_8UC3, testImg_8UC3_res, new_size, cv::INTER_NEAREST);
    resize_custom(testImg_16UC1, testImg_16UC1_res, new_size, cv::INTER_NEAREST);
    resize_custom(testImg_16UC3, testImg_16UC3_res, new_size, cv::INTER_NEAREST);
    resize_custom(testImg_32FC1, testImg_32FC1_res, new_size, cv::INTER_NEAREST);
    resize_custom(testImg_32FC3, testImg_32FC3_res, new_size, cv::INTER_NEAREST);
    cout << "Multi-type test passed" << endl;
}
