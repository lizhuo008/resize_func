#include <iostream>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "imgproc.hpp"
#include <immintrin.h>

using namespace std;

template <typename T>
resizeNNInvoker_custom<T>::resizeNNInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x_ofs, double _ify)
: input(_input), output(_output), inp_size(_inp_size), out_size(_out_size), x_ofs(_x_ofs), ify(_ify)
{

}

template <typename T>
void resizeNNInvoker_custom<T>::operator()(const cv::Range& range) const 
{
    int channels = input.channels();
    for (int y = range.start; y < range.end; y++)
    {
        T* D = output.ptr<T>(y);
        const T* S = input.ptr<T>( min( (int)floor(y * ify), inp_size.height - 1) );

        for(int x = 0; x < out_size.width; x++, D += channels)
        {
            const T* _tS = S + x_ofs[x];
            for (int k = 0; k < channels; k++)
                D[k] = _tS[k];
        }         
    }
}

void resizeNN_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{
#if (defined(USE_AVX2) && !defined(TEST))
    int channels = input.channels();
    int* x_ofs = static_cast<int*>(_mm_malloc(((out_size.width + 7) & -8) * sizeof(int), 32));
    if (x_ofs == nullptr)
        throw std::runtime_error("Failed to allocate memory for x_ofs");

    for(int x = 0; x < out_size.width; x += 8)
    {
        __m256i indices = _mm256_setr_epi32(x, x+1, x+2, x+3, x+4, x+5, x+6, x+7);
        __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(indices), _mm256_set1_ps(ifx));
        __m256i sx = _mm256_cvtps_epi32(scaled);

        __m256i max_width = _mm256_set1_epi32(inp_size.width - 1);
        sx = _mm256_min_epi32(sx, max_width);

        sx = _mm256_mullo_epi32(sx, _mm256_set1_epi32(channels));
        
        _mm256_store_si256((__m256i*)(x_ofs + x), sx);
    }

    for(int x = (out_size.width - out_size.width % 8); x < out_size.width; x++)
    {
        int sx = floor(x * ifx);
        x_ofs[x] = min(sx, inp_size.width - 1);
    }

    cv::Range range(0, out_size.height);

    switch (input.type())
    {
        case CV_8UC1:
        case CV_8UC3:
            cv::parallel_for_(range, simd::resizeNNInvoker_AVX2<uint8_t>(input, output, inp_size, out_size, x_ofs, ify));
            break;
        case CV_16UC1:
        case CV_16UC3:
            cv::parallel_for_(range, simd::resizeNNInvoker_AVX2<uint16_t>(input, output, inp_size, out_size, x_ofs, ify));
            break;
        case CV_32FC1:
        case CV_32FC3:
            cv::parallel_for_(range, simd::resizeNNInvoker_AVX2<float>(input, output, inp_size, out_size, x_ofs, ify));
            break;
        default:
            throw std::runtime_error("Unsupported image type");
    }

    _mm_free(x_ofs);
#else
    cv::AutoBuffer<int> _x_ofs(out_size.width);
    int* x_ofs = _x_ofs.data();
    int channels = input.channels();

    for(int x = 0; x < out_size.width; x++)
    {
        int sx = floor(x * ifx);
        x_ofs[x] = min(sx, inp_size.width - 1) * channels;
    }

    cv::Range range(0, out_size.height);

    switch (input.type())
    {
        case CV_8UC1:
        case CV_8UC3:
            cv::parallel_for_(range, resizeNNInvoker_custom<uchar>(input, output, inp_size, out_size, x_ofs, ify));
            break;
        case CV_16UC1:
        case CV_16UC3:
            cv::parallel_for_(range, resizeNNInvoker_custom<uint16_t>(input, output, inp_size, out_size, x_ofs, ify));
            break;
        case CV_32FC1:
        case CV_32FC3:
            cv::parallel_for_(range, resizeNNInvoker_custom<float>(input, output, inp_size, out_size, x_ofs, ify));
            break;
        default:
            throw std::runtime_error("Unsupported image type");
    }
#endif
}

template <typename T>
resizeBilinearInvoker_custom<T>::resizeBilinearInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x1_ofs, int* _x2_ofs, double* _wx2, double _ify)
: input(_input), output(_output), inp_size(_inp_size), out_size(_out_size), x1_ofs(_x1_ofs), x2_ofs(_x2_ofs), wx2(_wx2), ify(_ify)
{

}

template <typename T>
void resizeBilinearInvoker_custom<T>::operator()(const cv::Range& range) const
{
    int channels = input.channels();
    for (int y = range.start; y < range.end; y++)
    {   
        double src_y = (y + 0.5) * ify - 0.5;
        int y1 = static_cast<int>(src_y);
        int y2 = min(y1 + 1, inp_size.height - 1);
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

void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
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
        int x2 = min(x1 + 1, inp_size.width - 1); 
        x1_ofs[x] = x1 * channels;
        x2_ofs[x] = x2 * channels;
        wx2[x] = src_x - x1;
    }

    cv::Range range(0, out_size.height);

    switch (input.type())
    {
        case CV_8UC1:
        case CV_8UC3:
            cv::parallel_for_(range, resizeBilinearInvoker_custom<uchar>(input, output, inp_size, out_size, x1_ofs, x2_ofs, wx2, ify));
            break;
        case CV_16UC1:
        case CV_16UC3:
            cv::parallel_for_(range, resizeBilinearInvoker_custom<uint16_t>(input, output, inp_size, out_size, x1_ofs, x2_ofs, wx2, ify));
            break;  
        case CV_32FC1:
        case CV_32FC3:
            cv::parallel_for_(range, resizeBilinearInvoker_custom<float>(input, output, inp_size, out_size, x1_ofs, x2_ofs, wx2, ify));
            break;
        default:
            throw std::runtime_error("Unsupported image type");
    }
}

void resize_naive(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation)
{
    output = cv::Mat(new_size, input.type());
    cv::Size input_size = input.size();
    double ifx = (double)input_size.width / new_size.width;
    double ify = (double)input_size.height / new_size.height;

    switch (interpolation)
    {
        case cv::INTER_NEAREST:
            switch (input.type())
            {
                case CV_8UC1:
                    resizeNN_naive<uint8_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_8UC3:
                    resizeNN_naive<uint8_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_16UC1:
                    resizeNN_naive<uint16_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_16UC3:
                    resizeNN_naive<uint16_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_32FC1:
                    resizeNN_naive<float>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_32FC3:
                    resizeNN_naive<float>(input, output, input_size, new_size, ifx, ify);
                    break;
            }
            break;
        case cv::INTER_LINEAR:
            switch (input.type())
            {
                case CV_8UC1:
                    resizeBilinear_naive<uint8_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_8UC3:
                    resizeBilinear_naive<uint8_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_16UC1:
                    resizeBilinear_naive<uint16_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_16UC3:
                    resizeBilinear_naive<uint16_t>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_32FC1:
                    resizeBilinear_naive<float>(input, output, input_size, new_size, ifx, ify);
                    break;
                case CV_32FC3:
                    resizeBilinear_naive<float>(input, output, input_size, new_size, ifx, ify);
                    break;
            }
            break;
        default:
            std::cerr << "Interpolation method not implemented yet" << std::endl;
    }
}

void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation)
{   
    output = cv::Mat(new_size, input.type());
    cv::Size input_size = input.size();
    double ifx = (double)input_size.width / new_size.width;
    double ify = (double)input_size.height / new_size.height;

    switch (interpolation)
    {   
        case cv::INTER_NEAREST:
            resizeNN_custom(input, output, input_size, new_size, ifx, ify);
            break;
        case cv::INTER_LINEAR:
            resizeBilinear_custom(input, output, input_size, new_size, ifx, ify);
            break;
        default:
            std::cerr << "Interpolation method not implemented yet" << std::endl;
    }
}

