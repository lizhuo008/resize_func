#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>

using namespace std;

void compare_resize_runtime(const cv::Mat& input, const cv::Size& new_size, int interpolation)
{
    // Create output Mat objects for both methods
    cv::Mat output_custom(new_size, input.type());
    cv::Mat output_opencv(new_size, input.type());

    // Measure runtime of custom resize function
    auto start_custom = std::chrono::high_resolution_clock::now();
    resize_custom(input, output_custom, new_size, interpolation);  // Custom resize
    auto end_custom = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_custom = end_custom - start_custom;
    std::cout << "Custom resize runtime: " << duration_custom.count() << " seconds" << std::endl;

    // Measure runtime of OpenCV's default resize function
    auto start_opencv = std::chrono::high_resolution_clock::now();
    cv::resize(input, output_opencv, new_size, 0, 0, interpolation);  // OpenCV resize
    auto end_opencv = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_opencv = end_opencv - start_opencv;
    std::cout << "OpenCV resize runtime: " << duration_opencv.count() << " seconds" << std::endl;
}

int main(void)
{   
    // custom parameters
    string inp_path = "../samples/grayscale1.jpg";
    // string inp_path = "../samples/RGB1.jpg";
    // string out_path = "output.jpg";
    cv::Size new_size(10, 50);
    int interpolation = cv::INTER_LINEAR;

    cv::Mat input = cv::imread(inp_path);
    compare_resize_runtime(input, new_size, interpolation);

    // cv::Mat input = cv::imread(inp_path);
    // assert(!input.empty());
    // cv::Mat output = cv::Mat::zeros(new_size, input.type());
    
    // resize_custom(input, output, new_size, interpolation);

    // cv::imwrite(out_path, output);
    // cv::imshow("Resized Image", output);
    // cv::waitKey(0);
    // cv::destroyAllWindows(); // close all windows, is it necessary?

    return 0;
}