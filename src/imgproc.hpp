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

void resizeNN_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);

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

// template <typename T>
// void resizeNN_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);

template <typename T>
void resizeNN_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{
    cv::AutoBuffer<int> _x_ofs(out_size.width);
    int* x_ofs = _x_ofs.data();
    int channels = input.channels();

    for(int x = 0; x < out_size.width; x++)
    {
        int sx = floor(x * ifx);
        x_ofs[x] = std::min(sx, inp_size.width - 1) * channels;
    }

    for(int y = 0; y < out_size.height; y++)
    {
        T* D = output.ptr<T>(y);
        const T* S = input.ptr<T>(std::min((int)(floor(y * ify)), inp_size.height - 1));

        for(int x = 0; x < out_size.width; x++, D += channels)
        {
            const T* _tS = S + x_ofs[x];
            for(int k = 0; k < channels; k++)
                D[k] = _tS[k];
        }
    }
}

void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);

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
    void resizeNN_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);
}

template class simd::resizeNNInvoker_AVX2<uint8_t>;
template class simd::resizeNNInvoker_AVX2<uint16_t>;
template class simd::resizeNNInvoker_AVX2<float>;

#endif