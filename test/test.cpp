#if defined(TEST)
#include <iostream> 
#include <opencv2/opencv.hpp>
#include "../src/imgproc.hpp"
#include "test.hpp"
#include "../src/utils.hpp"
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

void multi_type_test(int interpolation)
{
    cv::Mat testImg_8UC1, testImg_8UC3, testImg_16UC1, testImg_16UC3, testImg_32FC1, testImg_32FC3;
    cv::Mat testImg_8UC1_res, testImg_8UC3_res, testImg_16UC1_res, testImg_16UC3_res, testImg_32FC1_res, testImg_32FC3_res;

    testImg_8UC1 = cv::imread("../samples/RGB1.jpg");
    testImg_8UC3 = cv::imread("../samples/RGB1.jpg");
    testImg_16UC1 = cv::imread("../samples/RGB1.jpg");
    testImg_16UC3 = cv::imread("../samples/RGB1.jpg");
    testImg_32FC1 = cv::imread("../samples/RGB1.jpg");
    testImg_32FC3 = cv::imread("../samples/RGB1.jpg");

    cv::Size new_size(1024, 1024);

    cout << "8UC1 test..." << endl;
    CVT_3C21C(testImg_8UC1);
    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(testImg_8UC1, testImg_8UC1_res, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(testImg_8UC1, testImg_8UC1_res, new_size, cv::INTER_LINEAR);
    }
    cv::imshow("8UC1 Image", testImg_8UC1_res);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "8UC3 test..." << endl;
    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(testImg_8UC3, testImg_8UC3_res, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(testImg_8UC3, testImg_8UC3_res, new_size, cv::INTER_LINEAR);
    }
    cv::imshow("8UC3 Image", testImg_8UC3_res);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "16UC1 test..." << endl;
    CVT_3C21C(testImg_16UC1);
    CVT_8U216U(testImg_16UC1);
    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(testImg_16UC1, testImg_16UC1_res, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(testImg_16UC1, testImg_16UC1_res, new_size, cv::INTER_LINEAR);
    }
    CVT_16U28U(testImg_16UC1_res);
    cv::imshow("16UC1 Image", testImg_16UC1_res);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "16UC3 test..." << endl;
    CVT_8U216U(testImg_16UC3);
    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(testImg_16UC3, testImg_16UC3_res, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(testImg_16UC3, testImg_16UC3_res, new_size, cv::INTER_LINEAR);
    }
    CVT_16U28U(testImg_16UC3_res);
    cv::imshow("16UC3 Image", testImg_16UC3_res);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "32FC1 test..." << endl;
    CVT_3C21C(testImg_32FC1);
    CVT_8U232F(testImg_32FC1);
    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(testImg_32FC1, testImg_32FC1_res, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(testImg_32FC1, testImg_32FC1_res, new_size, cv::INTER_LINEAR);
    }
    CVT_32F28U(testImg_32FC1_res);
    cv::imshow("32FC1 Image", testImg_32FC1_res);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "32FC3 test..." << endl;
    CVT_8U232F(testImg_32FC3);
    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(testImg_32FC3, testImg_32FC3_res, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(testImg_32FC3, testImg_32FC3_res, new_size, cv::INTER_LINEAR);
    }
    CVT_32F28U(testImg_32FC3_res);
    cv::imshow("32FC3 Image", testImg_32FC3_res);

    cv::waitKey(0);
    cv::destroyAllWindows();

    cout << "Multi-type test passed" << endl;
}

void amp_shr_test(int interpolation)
{
    cv::Mat testImg, testImg_res;
    testImg = cv::imread("../samples/RGB1.jpg");
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
    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(testImg, testImg_res, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(testImg, testImg_res, new_size, cv::INTER_LINEAR);
    }

    cv::imshow("Resized Image", testImg_res);

    std::cout << "Press any key to exit amplify/shrink test...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void multithread_test(int interpolation)
{
    int test_times = 200;
    cv::Size size(1024, 1024);

    cv::Mat testImg_8UC1, testImg_8UC3, testImg_16UC1, testImg_16UC3, testImg_32FC1, testImg_32FC3;
    cv::Mat testImg_8UC1_resp, testImg_8UC3_resp, testImg_16UC1_resp, testImg_16UC3_resp, testImg_32FC1_resp, testImg_32FC3_resp;
    cv::Mat testImg_8UC1_resn = cv::Mat::zeros(size, CV_8UC1);
    cv::Mat testImg_8UC3_resn = cv::Mat::zeros(size, CV_8UC3);
    cv::Mat testImg_16UC1_resn = cv::Mat::zeros(size, CV_16UC1);
    cv::Mat testImg_16UC3_resn = cv::Mat::zeros(size, CV_16UC3);
    cv::Mat testImg_32FC1_resn = cv::Mat::zeros(size, CV_32FC1);
    cv::Mat testImg_32FC3_resn = cv::Mat::zeros(size, CV_32FC3);

    createTestImage(testImg_8UC1, size, CV_8UC1);
    createTestImage(testImg_8UC3, size, CV_8UC3);
    createTestImage(testImg_16UC1, size, CV_16UC1);
    createTestImage(testImg_16UC3, size, CV_16UC3);
    createTestImage(testImg_32FC1, size, CV_32FC1);
    createTestImage(testImg_32FC3, size, CV_32FC3);

    cv::Size new_size(512, 512);

    double naiveTime = 0;
    double parallelTime = 0;
    for (size_t i = 0; i < test_times; i++)
    {   
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_naive(testImg_8UC1, testImg_8UC1_resn, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Naive");

        naiveTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_8UC1, testImg_8UC1_resp, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Parallel");

        parallelTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    cout << "Naive Resize Time for 8UC1: " << naiveTime / test_times << " ms\n";
    cout << "Parallel Resize Time for 8UC1: " << parallelTime / test_times << " ms\n";

    naiveTime = 0;
    parallelTime = 0;
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_naive(testImg_8UC3, testImg_8UC3_resn, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Naive");

        naiveTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_8UC3, testImg_8UC3_resp, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Parallel");

        parallelTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    cout << "Naive Resize Time for 8UC3: " << naiveTime / test_times << " ms\n";
    cout << "Parallel Resize Time for 8UC3: " << parallelTime / test_times << " ms\n";

    naiveTime = 0;
    parallelTime = 0;
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_naive(testImg_16UC1, testImg_16UC1_resn, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Naive");

        naiveTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_16UC1, testImg_16UC1_resp, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Parallel");

        parallelTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    cout << "Naive Resize Time for 16UC1: " << naiveTime / test_times << " ms\n";
    cout << "Parallel Resize Time for 16UC1: " << parallelTime / test_times << " ms\n";
    
    naiveTime = 0;
    parallelTime = 0;
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_naive(testImg_16UC3, testImg_16UC3_resn, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Naive");

        naiveTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_16UC3, testImg_16UC3_resp, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Parallel");

        parallelTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    cout << "Naive Resize Time for 16UC3: " << naiveTime / test_times << " ms\n";
    cout << "Parallel Resize Time for 16UC3 " << parallelTime / test_times << " ms\n";
    
    naiveTime = 0;
    parallelTime = 0;
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_naive(testImg_32FC1, testImg_32FC1_resn, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Naive");

        naiveTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_32FC1, testImg_32FC1_resp, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Parallel");

        parallelTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    cout << "Naive Resize Time for 32FC1: " << naiveTime / test_times << " ms\n";
    cout << "Parallel Resize Time for 32FC1: " << parallelTime / test_times << " ms\n";
    
    naiveTime = 0;
    parallelTime = 0;
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_naive(testImg_32FC3, testImg_32FC3_resn, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Naive");

        naiveTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (size_t i = 0; i < test_times; i++)
    {
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_32FC3, testImg_32FC3_resp, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Parallel");

        parallelTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    cout << "Naive Resize Time for 32FC3: " << naiveTime / test_times << " ms\n";
    cout << "Parallel Resize Time for 32FC3: " << parallelTime / test_times << " ms\n";

}

void measurePerformance(const cv::Size& input_size, const cv::Size& new_size, int dtype, int interpolation) 
{
    double customTime = 0;
    double openCVTime = 0;
    int test_times = 200;

    cv::Mat myOutput;
    cv::Mat input;
   
    if (interpolation == cv::INTER_NEAREST)
    {
        for (size_t i = 0; i < test_times; i++)
        {
            createTestImage(input, input_size, dtype, i);
            myOutput = cv::Mat::zeros(new_size, dtype);
            // TIME_START; 
            auto start_time = std::chrono::high_resolution_clock::now();
            simd::resize_AVX2(input, myOutput, new_size, cv::INTER_NEAREST);
            auto end_time = std::chrono::high_resolution_clock::now();
            // TIME_END("Custom");
            customTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }
    }else{
        for (size_t i = 0; i < test_times; i++)
        {
            createTestImage(input, input_size, dtype, i);
            myOutput = cv::Mat::zeros(new_size, dtype);
            // TIME_START;
            auto start_time = std::chrono::high_resolution_clock::now();
            resize_custom(input, myOutput, new_size, cv::INTER_LINEAR);
            auto end_time = std::chrono::high_resolution_clock::now();
            // TIME_END("Custom");
            customTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }
    }

    cv::Mat openCVOutput;
    for (size_t i = 0; i < test_times; i++)
    {
        createTestImage(input, input_size, dtype, i);
        openCVOutput = cv::Mat::zeros(new_size, dtype);
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        cv::resize(input, openCVOutput, new_size, interpolation);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("OpenCV");

        openCVTime += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    
    cout << "Custom Resize Time: " << customTime / test_times << " ms\n";
    cout << "OpenCV Resize Time: " << openCVTime / test_times << " ms\n";
}

void measure_accuracy(const cv::Size& input_size, const cv::Size& new_size, int dtype, int interpolation)
{
    cv::Mat input;
    cv::Mat output_custom;
    cv::Mat output_openCV;

    createTestImage(input, input_size, dtype, 0);
    output_custom = cv::Mat::zeros(new_size, dtype);
    output_openCV = cv::Mat::zeros(new_size, dtype);

    if (interpolation == cv::INTER_NEAREST)
    {
        simd::resize_AVX2(input, output_custom, new_size, cv::INTER_NEAREST);
        cv::resize(input, output_openCV, new_size, cv::INTER_NEAREST);
    }
    else
    {
        resize_custom(input, output_custom, new_size, cv::INTER_LINEAR);
        cv::resize(input, output_openCV, new_size, cv::INTER_LINEAR);
    }

    double diff = cv::norm(output_custom, output_openCV, cv::NORM_L2);
    double diff_percentage = diff / cv::norm(output_custom, cv::NORM_L2) * 100;

    cout << "Diff: " << diff << endl;
    cout << "Diff Percentage: " << diff_percentage << "%" << endl;

}

void standard_comp_test(int interpolation)
{
    cv::Mat testImg;
    std::vector<cv::Size> input_sizes = {
        cv::Size(256, 256),   // Small
        cv::Size(1024, 1024), // Medium
        cv::Size(2048, 2048)  // Large
    };
    
    std::vector<cv::Size> output_sizes = {
        cv::Size(1024, 1024), // Large output
        cv::Size(512, 512)    // Small output
    };
    
    std::vector<int> data_types = {
        CV_8UC1, CV_8UC3, 
        CV_16UC1, CV_16UC3, 
        CV_32FC1, CV_32FC3
    };

    for (int dtype : data_types) {
        for (const auto& input_size : input_sizes) {
            for (const auto& output_size : output_sizes) {
                std::cout << "Testing with:"
                         << " Input size: " << input_size
                         << " Output size: " << output_size
                         << " Data type: " << dtype 
                         << " Interpolation: " << interpolation << endl;
                measurePerformance(input_size, output_size, dtype, interpolation);
                measure_accuracy(input_size, output_size, dtype, interpolation);
            }
        }
    }
}

void simd_test()
{
    int test_times = 200;
    cv::Size size(512, 512);
    cv::Size new_size(1024, 1024);
    double ifx = 1.0 * new_size.width / size.width;
    double ify = 1.0 * new_size.height / size.height;
//8U
    double duration_simd_8UC1 = 0;
    double duration_custom_8UC1 = 0;
    double duration_simd_8UC3 = 0;
    double duration_custom_8UC3 = 0;
    
    cv::Mat testImg_8UC1, testImg_8UC1_res_custom, testImg_8UC1_res_simd, testImg_8UC3, testImg_8UC3_res_custom, testImg_8UC3_res_simd;
    
    for (int i = 0; i < test_times; i++)
    {   
        createTestImage(testImg_8UC1, size, CV_8UC1, i);
        testImg_8UC1_res_simd = cv::Mat::zeros(new_size, testImg_8UC1.type());
        
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        simd::resize_AVX2(testImg_8UC1, testImg_8UC1_res_simd, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("SIMD");

        duration_simd_8UC1+= std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_8UC1, size, CV_8UC1, i);
        testImg_8UC1_res_custom = cv::Mat::zeros(new_size, testImg_8UC1.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_8UC1, testImg_8UC1_res_custom, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Custom");

        duration_custom_8UC1 += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_8UC3, size, CV_8UC3, i);

        testImg_8UC3_res_simd = cv::Mat::zeros(new_size, testImg_8UC3.type());
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        simd::resize_AVX2(testImg_8UC3, testImg_8UC3_res_simd, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("SIMD");

        duration_simd_8UC3+= std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_8UC3, size, CV_8UC3, i);

        testImg_8UC3_res_custom = cv::Mat::zeros(new_size, testImg_8UC3.type());
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_8UC3, testImg_8UC3_res_custom, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Custom");

        duration_custom_8UC3 += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    cout << "AVG SIMD 8UC1 time: " << duration_simd_8UC1 / test_times << "ms" << endl;
    cout << "AVG Custom 8UC1 time: " << duration_custom_8UC1 / test_times << "ms" << endl;
    cout << "AVG SIMD 8UC3 time: " << duration_simd_8UC3 / test_times << "ms" << endl;
    cout << "AVG Custom 8UC3 time: " << duration_custom_8UC3 / test_times << "ms" << endl;

// 16U
    double duration_simd_16UC1 = 0;
    double duration_custom_16UC1 = 0;
    double duration_simd_16UC3 = 0;
    double duration_custom_16UC3 = 0;
    cv::Mat testImg_16UC1, testImg_16UC1_res_custom, testImg_16UC1_res_simd, testImg_16UC3, testImg_16UC3_res_custom, testImg_16UC3_res_simd;
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_16UC1, size, CV_16UC1, i);
        testImg_16UC1_res_simd = cv::Mat::zeros(new_size, testImg_16UC1.type());
        
        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        simd::resize_AVX2(testImg_16UC1, testImg_16UC1_res_simd, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("SIMD");

        duration_simd_16UC1+= std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_16UC1, size, CV_16UC1, i);
        testImg_16UC1_res_custom = cv::Mat::zeros(new_size, testImg_16UC1.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_16UC1, testImg_16UC1_res_custom, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Custom");

        duration_custom_16UC1 += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_16UC3, size, CV_16UC3, i);
        testImg_16UC3_res_simd = cv::Mat::zeros(new_size, testImg_16UC3.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        simd::resize_AVX2(testImg_16UC3, testImg_16UC3_res_simd, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("SIMD");

        duration_simd_16UC3+= std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_16UC3, size, CV_16UC3, i);
        testImg_16UC3_res_custom = cv::Mat::zeros(new_size, testImg_16UC3.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_16UC3, testImg_16UC3_res_custom, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Custom");

        duration_custom_16UC3 += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    cout << "AVG SIMD 16UC1 time: " << duration_simd_16UC1 / test_times << "ms" << endl;
    cout << "AVG Custom 16UC1 time: " << duration_custom_16UC1 / test_times << "ms" << endl;
    cout << "AVG SIMD 16UC3 time: " << duration_simd_16UC3 / test_times << "ms" << endl;
    cout << "AVG Custom 16UC3 time: " << duration_custom_16UC3 / test_times << "ms" << endl;

// 32F
    double duration_simd_32FC1 = 0;
    double duration_custom_32FC1 = 0;
    double duration_simd_32FC3 = 0;
    double duration_custom_32FC3 = 0;
    cv::Mat testImg_32FC1, testImg_32FC1_res_custom, testImg_32FC1_res_simd, testImg_32FC3, testImg_32FC3_res_custom, testImg_32FC3_res_simd;
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_32FC1, size, CV_32FC1, i);
        testImg_32FC1_res_simd = cv::Mat::zeros(new_size, testImg_32FC1.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        simd::resize_AVX2(testImg_32FC1, testImg_32FC1_res_simd, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("SIMD");

        duration_simd_32FC1+= std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_32FC1, size, CV_32FC1, i);
        testImg_32FC1_res_custom = cv::Mat::zeros(new_size, testImg_32FC1.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_32FC1, testImg_32FC1_res_custom, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Custom");

        duration_custom_32FC1 += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_32FC3, size, CV_32FC3, i);
        testImg_32FC3_res_simd = cv::Mat::zeros(new_size, testImg_32FC3.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        simd::resize_AVX2(testImg_32FC3, testImg_32FC3_res_simd, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("SIMD");

        duration_simd_32FC3+= std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    for (int i = 0; i < test_times; i++){
        createTestImage(testImg_32FC3, size, CV_32FC3, i);
        testImg_32FC3_res_custom = cv::Mat::zeros(new_size, testImg_32FC3.type());

        // TIME_START;
        auto start_time = std::chrono::high_resolution_clock::now();
        resize_custom(testImg_32FC3, testImg_32FC3_res_custom, new_size, cv::INTER_NEAREST);
        auto end_time = std::chrono::high_resolution_clock::now();
        // TIME_END("Custom");

        duration_custom_32FC3 += std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    cout << "AVG SIMD 32FC1 time: " << duration_simd_32FC1 / test_times << "ms" << endl;
    cout << "AVG Custom 32FC1 time: " << duration_custom_32FC1 / test_times << "ms" << endl;
    cout << "AVG SIMD 32FC3 time: " << duration_simd_32FC3 / test_times << "ms" << endl;
    cout << "AVG Custom 32FC3 time: " << duration_custom_32FC3 / test_times << "ms" << endl;

}
#endif