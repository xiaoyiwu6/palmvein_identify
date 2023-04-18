#ifndef GABORMAIN_H
#define GABORMAIN_H
//#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

void Gabor_CR_real_imaginary_image(Mat a_timage_in, Mat &a_tConv_real, Mat &a_tConv_imag, int theat, double SD, double CF, int sub_region_length);//进行子区域gabor滤波
double Gabor_standard_deviation(Mat a_timage_in);//标准差 sigma
double Gabor_centeral_frequency(Mat a_timage_in);//中心频率 miu
int calculateOrientations(Mat a_timage_in, int theta_num);//回傳輸入圖片的最大角度
double Gabor_GenvXY(Mat a_timage_in, double SD, int X, int Y, int X0, int Y0);//公式部分中的g/sigma(x, y)
void Gabor_Kernel(Mat &a_tGabor_real,Mat &a_tGabor_imaginary,int Theta,double SD,double CF,int sub_region_length);//gabor filter生成器，size=(5, 5)
void Gabor_sub_region_parameter(Mat a_timage_in,int &main_orientation,double &SD_GR,double &center_frequency, int sub_region_counter, int theta_num);//产生子区域的参数，用于生成gabor
Mat Gabor_remove_DC( Mat a_timage_Gabor); //图片0均值化，让数据中心处于(0,0)

void feature_extract(const string& ROI_path);


#endif
