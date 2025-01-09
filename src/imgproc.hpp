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

template <typename T>
class resizeBilinearInvoker_custom : public cv::ParallelLoopBody
{
    public:
    resizeBilinearInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x1_ofs, int* _x2_ofs, double* _wx2, double _ify);

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE;

    private:
    const cv::Mat& input;
    cv::Mat& output;
    const cv::Size& inp_size;
    const cv::Size& out_size;
    int* x1_ofs;
    int* x2_ofs;
    double* wx2;
    double ify;
};

void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify);

template <typename T>
void resizeBilinear_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{
    cv::AutoBuffer<int> _x1_ofs(out_size.width);
    int* x1_ofs = _x1_ofs.data();
    cv::AutoBuffer<int> _x2_ofs(out_size.width);
    int* x2_ofs = _x2_ofs.data();
    cv::AutoBuffer<double> _wx2(out_size.width);
    double* wx2 = _wx2.data();

    int channels = input.channels();

    for(int x = 0; x < out_size.width; x++)
    {   
        double src_x = (x + 0.5) * ifx - 0.5;
        int x1 = static_cast<int>(src_x);   
        int x2 = std::min(x1 + 1, inp_size.width - 1); 
        x1_ofs[x] = x1 * channels;
        x2_ofs[x] = x2 * channels;
        wx2[x] = src_x - x1;
    }
    for (int y = 0; y < out_size.height; y++)
    {   
        double src_y = (y + 0.5) * ify - 0.5;
        int y1 = static_cast<int>(src_y);
        int y2 = std::min(y1 + 1, inp_size.height - 1);
        int wy2 = src_y - y1;

        T* D = output.ptr<T>(y);
        const T* S1 = input.ptr<T>(y1);
        const T* S2 = input.ptr<T>(y2);

        for(int x = 0; x < out_size.width; x++ , D += channels)
        {   
            const T* i11 = S1 + x1_ofs[x];
            const T* i12 = S2 + x1_ofs[x];
            const T* i21 = S1 + x2_ofs[x];
            const T* i22 = S2 + x2_ofs[x];

            for(int k = 0; k < channels; k++)
            {   
                double iy1 = (1 - wx2[x]) * i11[k] + wx2[x] * i21[k];
                double iy2 = (1 - wx2[x]) * i12[k] + wx2[x] * i22[k];
                D[k] = static_cast<T>((1 - wy2) * iy1 + wy2 * iy2);
            }
        }
    }
}
/*
simple resize function w/o any error checking, hardware acceleration, etc.
*/
void resize_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation = cv::INTER_NEAREST);

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

    void resize_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation = cv::INTER_NEAREST);
}

template class simd::resizeNNInvoker_AVX2<uint8_t>;
template class simd::resizeNNInvoker_AVX2<uint16_t>;
template class simd::resizeNNInvoker_AVX2<float>;

#endif