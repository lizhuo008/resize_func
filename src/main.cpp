#include <iostream>
#include <cassert>
#include <string>

#include <opencv2/opencv.hpp>
#include <imgproc.hpp>

using namespace std;
int main(void)
{   
    // custom parameters
    string inp_path = "../samples/grayscale1.jpg";
    // string inp_path = "../samples/RGB1.jpg";
    string out_path = "output.jpg";
    cv::Size new_size(10, 1000);
    int interpolation = cv::INTER_NEAREST;

    cv::Mat input = cv::imread(inp_path);
    assert(!input.empty());
    cv::Mat output = cv::Mat::zeros(new_size, input.type());
    
    resize_custom(input, output, new_size, interpolation);

    cv::imwrite(out_path, output);
    cv::imshow("Resized Image", output);
    cv::waitKey(0);
    cv::destroyAllWindows(); // close all windows, is it necessary?

    return 0;
}