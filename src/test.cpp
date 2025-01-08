#if defined(TEST)
#include <iostream> 
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>
#include <test.hpp>
#include <utils.hpp>
#include <chrono>
#include <string>

using namespace std;
using namespace cv;

void basic_test(){
    string inp_path = "../samples/RGB1.jpg";
    string out_path = "output.jpg";
    cv::Size new_size(1024, 1024);
    int interpolation = cv::INTER_NEAREST;

    cv::Mat input = cv::imread(inp_path);
    cv::Mat output = cv::Mat::zeros(new_size, input.type());

    cv::imshow("Original Image", input);

    cout << "Press any key to continue...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
   
    resize_custom(input, output, new_size, interpolation);

    cv::imwrite(out_path, output);
    cv::imshow("Resized Image", output);

    cout << "Press any key to exit basic test...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

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

void amp_shr_test()
{
    cv::Mat testImg, testImg_res;
    cv::Size size(1000, 1000);

    createTestImage(testImg, size, CV_8UC3);
    cv::imshow("Original Image", testImg);

    cout << "Press any key to continue...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();

    int len, wid;

    cout << "Input new lengh(default 1000): ";
    cin >> len;
    cout << "Input new width(default 1000): ";
    cin >> wid;

    cv::Size new_size(len, wid);
    resize_custom(testImg, testImg_res, new_size, cv::INTER_NEAREST);

    cv::imshow("Resized Image", testImg_res);

    std::cout << "Press any key to exit amplify/shrink test...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void multithread_test(){

}

void measurePerformance(const cv::Mat& input, const cv::Size& new_size) 
{
    double customTime = 0;
    double openCVTime = 0;
    int test_times = 100;

    cv::Mat myOutput;
    for (size_t i = 0; i < test_times; i++)
    {
        TIME_START;
        resize_custom(input, myOutput, new_size, cv::INTER_NEAREST);
        TIME_END("Custom");

        customTime += std::chrono::duration<double, std::milli>(end - start).count();
    }

    cv::Mat openCVOutput;
    for (size_t i = 0; i < test_times; i++)
    {
        TIME_START;
        cv::resize(input, openCVOutput, new_size, 0, 0, cv::INTER_NEAREST);
        TIME_END("OpenCV");

        openCVTime += std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    std::cout << "Custom Resize Time: " << customTime / test_times << " ms\n";
    std::cout << "OpenCV Resize Time: " << openCVTime / test_times << " ms\n";
}

void standard_comp_test()
{
    cv::Mat smallImg, midImg, largeImg;
    cv::Mat smallRes, midRes, largeRes;

    cv::Size S(256, 256);
    cv::Size M(1024, 1024);
    cv::Size L(2048, 2048);

    createTestImage(smallImg, S, CV_8UC1);
    createTestImage(midImg, M, CV_8UC1);
    createTestImage(largeImg, L, CV_8UC3);

    cv::Size new_size(512,512);
    measurePerformance(smallImg, new_size);
    measurePerformance(midImg, new_size);
    measurePerformance(largeImg, new_size);

}

void simd_test()
{
    double duration_simd_8UC1 = 0;
    double duration_custom_8UC1 = 0;
    double duration_simd_8UC3 = 0;
    double duration_custom_8UC3 = 0;
    int test_times = 100;
    cv::Mat testImg_8UC1, testImg_8UC1_res_custom, testImg_8UC1_res_simd, testImg_8UC3, testImg_8UC3_res_custom, testImg_8UC3_res_simd;
    cv::Size size(512, 512);
    cv::Size new_size(1024, 1024);
    double ifx = 1.0 * new_size.width / size.width;
    double ify = 1.0 * new_size.height / size.height;
    for (int i = 0; i < test_times; i++)
    {   
        createTestImage(testImg_8UC1, size, CV_8UC1, i);
        testImg_8UC1_res_simd = cv::Mat::zeros(new_size, testImg_8UC1.type());
        
        TIME_START;
        simd::resizeNN_AVX2(testImg_8UC1, testImg_8UC1_res_simd, testImg_8UC1.size(), new_size, ifx, ify);
        TIME_END("SIMD");

        duration_simd_8UC1+= std::chrono::duration<double, std::milli>(end - start).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_8UC1, size, CV_8UC1, i);
        testImg_8UC1_res_custom = cv::Mat::zeros(new_size, testImg_8UC1.type());

        TIME_START;
        resizeNN_custom(testImg_8UC1, testImg_8UC1_res_custom, testImg_8UC1.size(), new_size, ifx, ify);
        TIME_END("Custom");

        duration_custom_8UC1 += std::chrono::duration<double, std::milli>(end - start).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_8UC3, size, CV_8UC3, i);

        testImg_8UC3_res_simd = cv::Mat::zeros(new_size, testImg_8UC3.type());
        TIME_START;
        simd::resizeNN_AVX2(testImg_8UC3, testImg_8UC3_res_simd, testImg_8UC3.size(), new_size, ifx, ify);
        TIME_END("SIMD");

        duration_simd_8UC3+= std::chrono::duration<double, std::milli>(end - start).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_8UC3, size, CV_8UC3, i);

        testImg_8UC3_res_custom = cv::Mat::zeros(new_size, testImg_8UC3.type());
        TIME_START;
        resizeNN_custom(testImg_8UC3, testImg_8UC3_res_custom, testImg_8UC3.size(), new_size, ifx, ify);
        TIME_END("Custom");

        duration_custom_8UC3 += std::chrono::duration<double, std::milli>(end - start).count();
    }

    cout << "AVG SIMD 8UC1 time: " << duration_simd_8UC1 / test_times << "ms" << endl;
    cout << "AVG Custom 8UC1 time: " << duration_custom_8UC1 / test_times << "ms" << endl;
    cout << "AVG SIMD 8UC3 time: " << duration_simd_8UC3 / test_times << "ms" << endl;
    cout << "AVG Custom 8UC3 time: " << duration_custom_8UC3 / test_times << "ms" << endl;
}
#endif