#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <imgproc.hpp>

using namespace std;

resizeNNInvoker_custom::resizeNNInvoker_custom(const cv::Mat& _input, cv::Mat& _output, const cv::Size& _inp_size, const cv::Size& _out_size, int* _x_ofs, double _ify)
: input(_input), output(_output), inp_size(_inp_size), out_size(_out_size), x_ofs(_x_ofs), ify(_ify)
{

}

void resizeNNInvoker_custom::operator()(const cv::Range& range) const 
{
    int pix_size = input.elemSize();
    for (int y = range.start; y < range.end; y++)
    {
        uchar* D = output.data + output.step * y;
        const uchar* S = input.ptr( min( (int)floor(y * ify), inp_size.height - 1) );

        switch (pix_size) // switch pix_size for further optimization
        {
            case 1:
                for (int x = 0; x < out_size.width; x++) // no SIMD optimization
                {
                    D[x] = S[x_ofs[x]];
                }
                break;
            case 3:
                for (int x = 0; x < out_size.width; x++, D += 3)
                {
                    const uchar* _tS = S + x_ofs[x];
                    D[0] = _tS[0];
                    D[1] = _tS[1];
                    D[2] = _tS[2];
                }
                break;
            default:
                for(int x = 0; x < out_size.width; x++, D += pix_size )
                {
                    const uchar* _tS = S + x_ofs[x];
                    for (int k = 0; k < pix_size; k++)
                        D[k] = _tS[k];
                }         
        }
    }
}

void resizeBilinear_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size out_size, double ifx, double ify)
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

void resizeNN_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& inp_size, const cv::Size out_size, double ifx, double ify)
{
    cv::AutoBuffer<int> _x_ofs(out_size.width);
    int* x_ofs = _x_ofs.data();
    int pix_size = input.elemSize();

    for(int x = 0; x < out_size.width; x++)
    {
        int sx = floor(x * ifx);
        x_ofs[x] = min(sx, inp_size.width - 1)*pix_size;
    }

    cv::Range range(0, out_size.height);
    cv::parallel_for_(range, resizeNNInvoker_custom(input, output, inp_size, out_size, x_ofs, ify));
}

void resize_custom(const cv::Mat& input, cv::Mat& output, const cv::Size& new_size, int interpolation)
{
    cv::Size input_size = input.size();
    double ifx = (double)input_size.width / new_size.width;
    double ify = (double)input_size.height / new_size.height;

    if (interpolation == cv::INTER_NEAREST)
    {
        resizeNN_custom(input, output, input_size, new_size, ifx, ify);
    }
    else if (interpolation == cv::INTER_LINEAR)
    {
        resizeBilinear_custom(input, output, input_size, new_size, ifx, ify);
    }
    else
    {
        std::cerr << "Interpolation method not implemented yet" << std::endl;
    }
}