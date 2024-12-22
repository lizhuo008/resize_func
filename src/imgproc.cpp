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

        switch (pix_size)
        {
            case 1:
                for (int x = 0; x < out_size.width; x++)
                {
                    uchar t0 = S[x_ofs[x]];
                    uchar t1 = S[x_ofs[x+1]];
                    D[x] = t0;
                    D[x+1] = t1;
                }

                for (int x = 0; x < out_size.width; x++)
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
    else
    {
        std::cerr << "interpolation method haven't implemented yet" << std::endl;
    }
}