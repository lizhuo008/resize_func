#if defined(TEST)
#include <iostream> 
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>
#include <test.hpp>
#include <utils.hpp>
#include <chrono>

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