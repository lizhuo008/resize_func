#include <iostream>
#include <cassert>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "./src/imgproc.hpp"
#include "./test/test.hpp"
#include "./src/utils.hpp"

using namespace std;

int main(void)
{   

#if defined(TEST)
    // cout << "******************************************" << endl;
    // cout << "*************** Basic Test ***************" << endl;
    // cout << "******************************************" << endl;
    // basic_test();

    /*
    test multi type, amp shr, multithread, standard comp, simd can totally map the requirements

    Actually, Our method satisfies the basic interpolation requirements, INTER_NEAREST and INTER_LINEAR

    Both of them can meet all data types(channels and basic types), all input sizes, and all output sizes
    and Multithread is also supported

    Specialized SIMD is also supported, which can accelerate the performance of the INTER_NEAREST interpolation
     */
    cout << "***************************************************" << endl;
    cout << "******************* Multi Type Test ***************" << endl;
    cout << "***************************************************" << endl;
    multi_type_test(cv::INTER_NEAREST);
    multi_type_test(cv::INTER_LINEAR);
    cout << "***************************************************" << endl;
    cout << "******************* Amp Shr Test ******************" << endl;
    cout << "***************************************************" << endl;
    amp_shr_test(cv::INTER_NEAREST);
    amp_shr_test(cv::INTER_NEAREST);
    amp_shr_test(cv::INTER_LINEAR);
    amp_shr_test(cv::INTER_LINEAR);
    cout << "**************************************************" << endl;
    cout << "*************** Multithread Test *****************" << endl;
    cout << "**************************************************" << endl;
    multithread_test(cv::INTER_NEAREST);
    multithread_test(cv::INTER_LINEAR);
    cout << "**************************************************" << endl;
    cout << "******************* SIMD Test ********************" << endl;
    cout << "**************************************************" << endl;
    simd_test();
    cout << "**************************************************" << endl;
    cout << "*************** Standard Comp Test ***************" << endl;
    cout << "**************************************************" << endl;
    standard_comp_test(cv::INTER_NEAREST);
    standard_comp_test(cv::INTER_LINEAR);
    

    cout << "All tests passed" << endl;
#else
    // // custom parameters
    // // string inp_path = "../samples/grayscale1.jpg";
    // string inp_path = "../samples/RGB1.jpg"; // the path is relative to compile directory (build/), not the project root
    // string out_path = "output.jpg";
    // cv::Size new_size(1024, 1024);
    // // int interpolation = cv::INTER_LINEAR;
    // int interpolation = cv::INTER_NEAREST;

    // cv::Mat input = cv::imread(inp_path);
    // // CVT_3C21C(input);
    // // CVT_8U232F(input);
    // CVT_8U216U(input);

    // cout << "Input image type: " << input.type() << endl;
    
    // cv::Mat output = cv::Mat::zeros(new_size, input.type());

    // cout << "Output image step: " << output.step << endl;
    
    // double ifx = (double)input.size().width / new_size.width;
    // double ify = (double)input.size().height / new_size.height;
    // // resizeNN_naive<uint16_t>(input, output, input.size(), new_size, ifx, ify);
    // // resizeBilinear_naive<float>(input, output, input.size(), new_size, ifx, ify);
    // // resize_custom(input, output, new_size, cv::INTER_NEAREST);
    // resize_custom(input, output, new_size, cv::INTER_LINEAR);
    

    // cout << "Output image type: " << output.type() << endl;

    // // output.convertTo(output, CV_8U, 1.0/256.0);
    // // CVT_32F28U(output);
    // CVT_16U28U(output);
    // cv::imwrite(out_path, output);
    // cv::imshow("Resized Image", output);
    // cv::waitKey(0);
    // cv::destroyAllWindows(); // close all windows, is it necessary?
    cout << "Test mode is not enabled" << endl;

    string inp_path;
    cv::Size new_size;
    int interpolation;
    cout << "Please input the input image path: ";
    cin >> inp_path;
    cout << "Please input the new size (width height): ";
    cin >> new_size.width >> new_size.height;
    cout << "Please input the interpolation method: ";
    cin >> interpolation;

    cv::Mat input = cv::imread(inp_path);
    cv::Mat output = cv::Mat::zeros(new_size, input.type());
    resize_custom(input, output, new_size, interpolation);
    cv::imshow("Resized Image", output);

    cout << "Press any key to exit" << endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
#endif

    return 0;
}