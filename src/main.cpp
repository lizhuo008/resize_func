#include <iostream>
#include <cassert>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>
#include <test.hpp>
#include <utils.hpp>

using namespace std;

int main(void)
{   

#if defined(TEST)
    basic_test();
    // multi_type_test();
    // simd_test();
    amp_shr_test();
    standard_comp_test();

    cout << "All tests passed" << endl;
#else
    cout << "Test mode is not enabled" << endl;
#endif

    return 0;
}