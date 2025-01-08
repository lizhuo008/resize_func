#include <iostream>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>
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

//refactor later
void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{
    int pix_size = input.elemSize();
    
    // Loop over every pixel in the output image
    for (int y = 0; y < out_size.height; y++)
    {
        for (int x = 0; x < out_size.width; x++)
        {
            // Map the current pixel in the output image to the corresponding coordinates in the input image
            double src_x = x * ifx;
            double src_y = y * ify;

            int x1 = static_cast<int>(src_x);  // Floor of src_x
            int y1 = static_cast<int>(src_y);  // Floor of src_y

            int x2 = min(x1 + 1, inp_size.width - 1);  // Right neighbor
            int y2 = min(y1 + 1, inp_size.height - 1); // Bottom neighbor

            // Calculate interpolation weights
            double dx = src_x - x1;
            double dy = src_y - y1;

            // Get pixel values from the four neighboring pixels
            const uchar* p1 = input.ptr(y1) + x1 * pix_size;
            const uchar* p2 = input.ptr(y1) + x2 * pix_size;
            const uchar* p3 = input.ptr(y2) + x1 * pix_size;
            const uchar* p4 = input.ptr(y2) + x2 * pix_size;

            // Interpolate horizontally first
            double r1 = (1 - dx) * p1[0] + dx * p2[0]; // Red channel interpolation (horizontal)
            double g1 = (1 - dx) * p1[1] + dx * p2[1]; // Green channel interpolation (horizontal)
            double b1 = (1 - dx) * p1[2] + dx * p2[2]; // Blue channel interpolation (horizontal)

            double r2 = (1 - dx) * p3[0] + dx * p4[0]; // Red channel interpolation (horizontal)
            double g2 = (1 - dx) * p3[1] + dx * p4[1]; // Green channel interpolation (horizontal)
            double b2 = (1 - dx) * p3[2] + dx * p4[2]; // Blue channel interpolation (horizontal)

            // Interpolate vertically
            output.data[y * output.step + x * pix_size] = static_cast<uchar>((1 - dy) * r1 + dy * r2);  // Red channel
            output.data[y * output.step + x * pix_size + 1] = static_cast<uchar>((1 - dy) * g1 + dy * g2);  // Green channel
            output.data[y * output.step + x * pix_size + 2] = static_cast<uchar>((1 - dy) * b1 + dy * b2);  // Blue channel
        }
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

