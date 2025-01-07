#include <iostream>
#include <cassert>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>
#include <test.hpp>
#include <utils.hpp>

using namespace std;

int main(void)
{   

#if defined(TEST)

    multi_type_test();
    simd_test();

    cout << "All tests passed" << endl;
#else
    // custom parameters
    // string inp_path = "../samples/grayscale1.jpg";
    // // string inp_path = "../samples/RGB1.jpg";
    // string out_path = "output.jpg";
    // cv::Size new_size(1024, 1024);
    // // int interpolation = cv::INTER_LINEAR;
    // int interpolation = cv::INTER_NEAREST;

    // cv::Mat input = cv::imread(inp_path);
    // cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
    // cout << "Input image type: " << input.type() << endl;
    // // compare_resize_runtime(input, new_size, interpolation);

    // // cv::Mat input = cv::imread(inp_path);
    // // assert(!input.empty());
    // cv::Mat output = cv::Mat::zeros(new_size, input.type());
    
    // resize_custom(input, output, new_size, interpolation);
    // compare_resize_runtime(input, new_size, interpolation);

    // cv::imwrite(out_path, output);
    // cv::imshow("Resized Image", output);
    // cv::waitKey(0);
    // cv::destroyAllWindows(); // close all windows, is it necessary?
    cout << "Test mode is not enabled" << endl;
#endif

    return 0;
}