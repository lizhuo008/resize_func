#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <imgproc.hpp>
#include <immintrin.h>

using namespace std;

template <typename T>
simd::resizeNNInvoker_AVX2<T>::resizeNNInvoker_AVX2(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x_ofs, double _ify)
: input(_input), output(_output), inp_size(_inp_size), out_size(_out_size), x_ofs(_x_ofs), ify(_ify)
{

}

// template <typename T>
// void simd::resizeNNInvoker_AVX2<T>::operator()(const cv::Range& range) const 
// {   
//     const int avx_width = 32;
    
//     int width = out_size.width;
//     int channels = input.channels();

//     switch (channels)
//     {
//         case 1:
//         {   
//             for (int y = range.start; y < range.end; y++)
//             {
//                 T* D = output.ptr<T>(y);
//                 const T* S = input.ptr<T>( min( (int)floor(y * ify), inp_size.height - 1) );
//                 const __m256i SHUFFLE_MASK = _mm256_setr_epi8(
//                 0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
//                 0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
//                 const __m256i PERMUTE_MASK = _mm256_setr_epi32(0x0, 0x4, -1, -1, -1, -1, -1, -1);\
//                 const int safe_width = (width - width % 8) - 24;

//                 for (int x = 0; x < safe_width; x += 8)
//                 {
//                     __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
//                     __m256i src = _mm256_i32gather_epi32((int*)S, idx, 1);
//                     __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
//                     _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
//                 } 
//                 for (int x = safe_width; x < width; x++)
//                 {
//                     D[x] = S[x_ofs[x]];
//                 }
//             }
//             break;
//         }
//         case 3:
//         {
//             for (int y = range.start; y < range.end; y++)
//             { 
//                 T* D = output.ptr<T>(y);
//                 const T* S = input.ptr<T>( min( (int)floor(y * ify), inp_size.height - 1) );
//                 const __m256i SHUFFLE_MASK = _mm256_setr_epi8(
//                 0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
//                 0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);
//                 const __m256i PERMUTE_MASK = _mm256_setr_epi32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, -1, -1); 
//                 const int safe_width = (width - width % 8) - 8;

//                 for (int x = 0; x < safe_width; x += 8)
//                 {
//                     __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
//                     __m256i src = _mm256_i32gather_epi32((int*)S, idx, 1);
//                     __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
//                     _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
//                 } 
//                 T* _tD = D + safe_width * channels;
//                 for (int x = safe_width; x < width; x++, _tD += channels)
//                 {   
//                     const T* _tS = S + x_ofs[x];
//                     for (int k = 0; k < channels; k++)
//                         _tD[k] = _tS[k];
//                 }
//             }
//             break;
//         }
//         default:
//             throw std::runtime_error("Unsupported image type");
//     }
// }

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

                for (int x = 0; x < safe_width; x += 8)
                {
                    __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
                    __m256i src = _mm256_i32gather_epi32((int*)S, idx, 1);
                    __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
                    _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
                } 
                break;
            }
            case CV_16UC1:
            {   
                __m256i SHUFFLE_MASK = _mm256_setr_epi16(
                0x0, 0x2, 0x4, 0x6, -1, -1, -1, -1,
                0x0, 0x2, 0x4, 0x6, -1, -1, -1, -1);
                __m256i PERMUTE_MASK = _mm256_setr_epi32(0x0, 0x1, 0x4, 0x5, -1, -1, -1, -1); 
                safe_width = (width - width % 8) - 8;

                for (int x = 0; x < safe_width; x += 8)
                {
                    __m256i idx = _mm256_loadu_si256((__m256i*)(x_ofs + x));
                    __m256i src = _mm256_i32gather_epi32((int*)S, idx, 1);
                    __m256i src_perm = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(src, SHUFFLE_MASK), PERMUTE_MASK);
                    _mm256_storeu_si256((__m256i*)(D + x * channels), src_perm);
                }
                break;
            }
            case CV_16UC3:
            {   
                
                break;
            }
            case CV_32FC1:
            {
                break;
            }
            case CV_32FC3:
            {
                break;
            }
            default:
                throw std::runtime_error("Unsupported image type");
        }
        T* _tD = D + safe_width * channels;
        for (int x = safe_width; x < width; x++, _tD += channels)
        {   
            const T* _tS = S + x_ofs[x];
            for (int k = 0; k < channels; k++)
                _tD[k] = _tS[k];
        }
    }   
}

//TEST ONLY
void simd::resizeNN_AVX2(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size& out_size, double ifx, double ify)
{   
    int* x_ofs = static_cast<int*>(_mm_malloc(((out_size.width + 7) & -8) * sizeof(int), 32));
    if (x_ofs == nullptr)
        throw std::runtime_error("Failed to allocate memory for x_ofs");
    int channels = input.channels();
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
            cv::parallel_for_(range, resizeNNInvoker_AVX2<uint8_t>(input, output, inp_size, out_size, x_ofs, ify));
            break;
        case CV_16UC1:
        case CV_16UC3:
            //TODO: implement
            cout << "Not implemented" << endl;
            break;
        case CV_32FC1:
        case CV_32FC3:
            //TODO: implement
            cout << "Not implemented" << endl;
            break;
        default:
            throw std::runtime_error("Unsupported image type");
    }
    _mm_free(x_ofs);
}