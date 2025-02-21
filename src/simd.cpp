#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "imgproc.hpp"
#include <immintrin.h>

using namespace std;

template <typename T>
simd::resizeNNInvoker_AVX2<T>::resizeNNInvoker_AVX2(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x_ofs, double _ify)
: input(_input), output(_output), inp_size(_inp_size), out_size(_out_size), x_ofs(_x_ofs), ify(_ify)
{

}

template <typename T>
void simd::resizeNNInvoker_AVX2<T>::operator()(const cv::Range& range) const 
{   
    const int avx_width = 32;
    
    int width = out_size.width;
    int channels = input.channels();

    for (int y = range.start; y < range.end; y++) 
    {
        T* D = output.ptr<T>(y);
        const T* S = input.ptr<T>( min( (int)floor(y * ify), inp_size.height - 1) );

        __m256i SHUFFLE_MASK, PERMUTE_MASK;
        int safe_width;

        switch (input.type())
        {
            case CV_8UC1:
            {   
                const __m256i SHUFFLE_MASK = _mm256_setr_epi8(
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                const __m256i PERMUTE_MASK = _mm256_setr_epi32(0x0, 0x4, -1, -1, -1, -1, -1, -1);\
                safe_width = (width - width % 8) - 24;
                safe_width = safe_width >= 0 ? safe_width : 0;
                for (int x = 0; x < safe_width; x += 8)
                {
                    __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
                    __m256i src = _mm256_i32gather_epi32((int*)S, idx, 1);
                    __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
                    _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
                }
                break;
            }
            case CV_8UC3:
            {
                __m256i SHUFFLE_MASK = _mm256_setr_epi8(
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);
                __m256i PERMUTE_MASK = _mm256_setr_epi32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, -1, -1); 
                safe_width = (width - width % 8) - 8;
                safe_width = safe_width >= 0 ? safe_width : 0;
                for (int x = 0; x < safe_width; x += 8)
                {
                    __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
                    __m256i src = _mm256_i32gather_epi32((int*)S, idx, 1);
                    __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
                    _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
                } 
                break;
            }
            case CV_16UC1: // checked
            {   
                const __m256i SHUFFLE_MASK = _mm256_setr_epi8(
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1);
                const __m256i PERMUTE_MASK = _mm256_setr_epi32(0x0, 0x1, 0x4, 0x5, -1, -1, -1, -1);
                safe_width = (width - width % 8) - 8;
                safe_width = safe_width >= 0 ? safe_width : 0;

                for (int x = 0; x < safe_width; x += 8)
                {
                    __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
                    __m256i src = _mm256_i32gather_epi32((int*)S, idx, 2);
                    __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
                    _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
                }
                break;
            }
            case CV_16UC3:
            {   
                __m256i SHUFFLE_MASK = _mm256_setr_epi8(
                0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, -1, -1, -1, -1);
                __m256i PERMUTE_MASK = _mm256_setr_epi32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, -1, -1); 
                safe_width = (width - width % 4) - 4;
                safe_width = safe_width >= 0 ? safe_width : 0;
                __m128i all_ones = _mm_set1_epi32(0xFFFFFFFF); 

                for (int x = 0; x < safe_width; x += 4)
                {
                    __m128i idx = _mm_maskload_epi32((int*)(x_ofs + x), all_ones);
                    __m256i src = _mm256_i32gather_epi64((long long*)S, idx, 2);
                    __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
                    _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
                }
                break;
            }
            case CV_32FC1:
            {   
                safe_width = (width - width % 8);
                safe_width = safe_width >= 0 ? safe_width : 0;
                for (int x = 0; x < safe_width; x += 8)
                {
                    __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
                    __m256 src = _mm256_i32gather_ps((float*)S, idx, 4);
                    _mm256_storeu_ps((float*)(D + x * channels), src);
                }
                break;
            }
            case CV_32FC3:
            {   
                // safe_width = (width - width % 2) - 2;
                // __m128i all_ones = _mm_set1_epi32(0xFFFFFFFF); 

                // for (int x = 0; x < safe_width; x += 2)
                // {
                //     __m128 src_low = _mm_maskload_ps((float*)(S + x_ofs[x]), all_ones);
                //     __m128 src_high = _mm_maskload_ps((float*)(S + x_ofs[x + 1]), all_ones);
                //     _mm_storeu_ps((float*)(D + x * channels), src_low);
                //     _mm_storeu_ps((float*)(D + (x + 1) * channels), src_high);
                // }

                safe_width = (width * channels - (width * channels) % 8);
                safe_width = safe_width >= 0 ? safe_width : 0;
                
                for (int x = 0; x < safe_width; x += 8)
                {
                    __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
                    __m256 src = _mm256_i32gather_ps((float*)S, idx, 4);
                    _mm256_storeu_ps((float*)(D + x), src);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported image type");
        }
        if (input.type() != CV_32FC3)
        {
            T* _tD = D + safe_width * channels;
            for (int x = safe_width; x < width; x++, _tD += channels)
            {   
                const T* _tS = S + x_ofs[x];
                for (int k = 0; k < channels; k++)
                    _tD[k] = _tS[k];
            }
        }else{
            T* _tD = D + safe_width;
            for (int x = safe_width; x < width * channels; x++, _tD++)
            {   
                const T* _tS = S + x_ofs[x];
                _tD[0] = _tS[0];
            }
        }

    }   
}

//TEST ONLY
#if defined(TEST)
void simd::resizeNN_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{   
    int* x_ofs = static_cast<int*>(_mm_malloc(((out_size.width + 7) & -8) * sizeof(int), 32));
    if (x_ofs == nullptr)
        throw std::runtime_error("Failed to allocate memory for x_ofs");
    int channels = input.channels();
    int safe_bound = out_size.width - out_size.width % 8;
    safe_bound = safe_bound >= 0 ? safe_bound : 0;
    for(int x = 0; x < safe_bound; x += 8)
    {   
        __m256i indices = _mm256_setr_epi32(x, x+1, x+2, x+3, x+4, x+5, x+6, x+7);
        __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(indices), _mm256_set1_ps(ifx));
        __m256i sx = _mm256_cvtps_epi32(scaled);

        __m256i max_width = _mm256_set1_epi32(inp_size.width - 1);
        sx = _mm256_min_epi32(sx, max_width);

        sx = _mm256_mullo_epi32(sx, _mm256_set1_epi32(channels));
        
        _mm256_store_si256((__m256i*)(x_ofs + x), sx);
    }
    for(int x = safe_bound; x < out_size.width; x++)
    {   
        int sx = floor(x * ifx);
        x_ofs[x] = min(sx, inp_size.width - 1) * channels;
    }

    int* x_ofs_32F;
    if (input.type() == CV_32FC3)
    {   
        x_ofs_32F = static_cast<int*>(_mm_malloc(((out_size.width + 7) & -8) * sizeof(int) * channels, 32));
        for(int x = 0; x < out_size.width; x ++)
        {
            for(int k = 0; k < channels; k++)
                x_ofs_32F[x * channels + k] = x_ofs[x] + k;
        }
        _mm_free(x_ofs);
        x_ofs = x_ofs_32F;
    }

//more aggressive version get a little bit faster 
// if (input.type() == CV_32FC3)
// {
//     x_ofs_32F = static_cast<int*>(_mm_malloc(((out_size.width + 7) & -8) * sizeof(int) * channels, 32));

//     __m256i vChannels = _mm256_set1_epi32(channels); 
//     for (int x = 0; x < out_size.width; x += 8) 
//     {    
//         __m256i base = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x_ofs[x]));
//         for (int k = 0; k < channels; ++k) 
//         {
//             __m256i offset = _mm256_add_epi32(base, _mm256_set1_epi32(k));
//             _mm256_storeu_si256(reinterpret_cast<__m256i*>(&x_ofs_32F[x * channels + k * 8]), offset);
//         }
//     }
//     _mm_free(x_ofs);
//     x_ofs = x_ofs_32F;
// }

    
    cv::Range range(0, out_size.height);

    switch (input.type())
    {
        case CV_8UC1:
        case CV_8UC3:
            cv::parallel_for_(range, resizeNNInvoker_AVX2<uint8_t>(input, output, inp_size, out_size, x_ofs, ify), output.total()/(double)(1<<16));
            break;
        case CV_16UC1:
        case CV_16UC3:
            cv::parallel_for_(range, resizeNNInvoker_AVX2<uint16_t>(input, output, inp_size, out_size, x_ofs, ify), output.total()/(double)(1<<16));
            break;
        case CV_32FC1:
        case CV_32FC3:
            cv::parallel_for_(range, resizeNNInvoker_AVX2<float>(input, output, inp_size, out_size, x_ofs, ify), output.total()/(double)(1<<16));
            break;
        default:
            throw std::runtime_error("Unsupported image type");
    }
    _mm_free(x_ofs);
}

void simd::resize_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation)
{
    output = cv::Mat(new_size, input.type());
    cv::Size input_size = input.size();
    double ifx = (double)input_size.width / new_size.width;
    double ify = (double)input_size.height / new_size.height;

    switch (interpolation)
    {
        case cv::INTER_NEAREST:
            resizeNN_AVX2(input, output, input_size, new_size, ifx, ify);
            break;
        case cv::INTER_LINEAR:
            // resizeBilinear_AVX2(input, output, input_size, new_size, ifx, ify);
            break;
        default:
            std::cerr << "Interpolation method not implemented yet" << std::endl;
    }
}
#endif