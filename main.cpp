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
/*
for code and structure simplicity, we use the same code for both test and normal use

you can set the TEST macro to enable the test mode

a standard test is provided, which can test the performance of the interpolation methods
*/
#if defined(TEST)
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
    cout << "Test mode is not enabled" << endl;

    string inp_path;
    cv::Size new_size;
    int interpolation;
    cout << "Please input the input image path: ";
    cin >> inp_path;
    cout << "Please input the new size (width height): ";
    cin >> new_size.width >> new_size.height;
    cout << "Please input the interpolation method (0: Nearest Neighbor, 1: Bilinear): ";
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