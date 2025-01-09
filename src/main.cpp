#include <iostream>
#include <cassert>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "imgproc.hpp"
#include "../test/test.hpp"
#include "utils.hpp"

using namespace std;

int main(void)
{   

#if defined(TEST)
    // basic_test();
    // multi_type_test();
    // amp_shr_test();
    multithread_test();
    // standard_comp_test();
    // simd_test();

    cout << "All tests passed" << endl;
#else
    // custom parameters
    // string inp_path = "../samples/grayscale1.jpg";
    string inp_path = "../samples/RGB1.jpg";
    string out_path = "output.jpg";
    cv::Size new_size(1024, 1024);
    // int interpolation = cv::INTER_LINEAR;
    int interpolation = cv::INTER_NEAREST;

    cv::Mat input = cv::imread(inp_path);
    // CVT_3C21C(input);
    // CVT_8U232F(input);
    CVT_8U216U(input);

    cout << "Input image type: " << input.type() << endl;
    
    cv::Mat output = cv::Mat::zeros(new_size, input.type());

    cout << "Output image step: " << output.step << endl;
    
    double ifx = (double)input.size().width / new_size.width;
    double ify = (double)input.size().height / new_size.height;
    // resizeNN_naive<uint16_t>(input, output, input.size(), new_size, ifx, ify);
    // resizeBilinear_naive<float>(input, output, input.size(), new_size, ifx, ify);
    // resize_custom(input, output, new_size, cv::INTER_NEAREST);
    resize_custom(input, output, new_size, cv::INTER_LINEAR);
    

    cout << "Output image type: " << output.type() << endl;

    // output.convertTo(output, CV_8U, 1.0/256.0);
    // CVT_32F28U(output);
    CVT_16U28U(output);
    cv::imwrite(out_path, output);
    cv::imshow("Resized Image", output);
    cv::waitKey(0);
    cv::destroyAllWindows(); // close all windows, is it necessary?
    cout << "Test mode is not enabled" << endl;
#endif

    return 0;
}