#ifndef IMGPROC_HPP
#define IMGPROC_HPP

#include <opencv2/opencv.hpp>

/*
Inherit from cv::ParallelLoopBody to use parallel_for_ function
overload the operator() function to perform the actual operation
*/
template <typename T>
class resizeNNInvoker_custom : public cv::ParallelLoopBody
{
    public:
    resizeNNInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* x_ofs, double _ify);

    void operator()(const cv::Range& range) const CV_OVERRIDE;

    private:
    const cv::Mat& input;
    cv::Mat& output;
    const cv::Size& inp_size;
    const cv::Size& out_size;
    int* x_ofs;
    double ify;
};

void resizeNN_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size out_size, double ifx, double ify);

class resizeBilinearInvoker_custom : public cv::ParallelLoopBody
{
    public:
    resizeBilinearInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, double _ifx, double _ify);

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE;

    private:
    const cv::Mat& input;
    cv::Mat& output;
    const cv::Size& inp_size;
    const cv::Size& out_size;
    double ifx;
    double ify;
};

void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size out_size, double ifx, double ify);

/*
simple resize function w/o any error checking, hardware acceleration, etc.
*/
void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation = cv::INTER_NEAREST);

namespace simd
{
    template <typename T>
    class resizeNNInvoker_AVX2 : public cv::ParallelLoopBody
    {
        public:
        resizeNNInvoker_AVX2(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* x_ofs, double _ify);

        void operator()(const cv::Range& range) const CV_OVERRIDE;

        private:
        const cv::Mat& input;
        cv::Mat& output;
        const cv::Size& inp_size;
        const cv::Size& out_size;
        int* x_ofs;
        double ify;
    };
    void resizeNN_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size out_size, double ifx, double ify);
}

#endif