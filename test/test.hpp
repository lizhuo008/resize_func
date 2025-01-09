#ifndef TEST_HPP
#define TEST_HPP

void basic_test();

void multi_type_test(int interpolation = cv::INTER_NEAREST);

void amp_shr_test(int interpolation = cv::INTER_NEAREST);

void multithread_test(int interpolation = cv::INTER_NEAREST);

void standard_comp_test(int interpolation = cv::INTER_NEAREST);

void simd_test();

#endif