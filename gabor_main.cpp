#include "gabor_main.h"


double Gabor_standard_deviation(Mat a_timage_in)//評估標準差
{
	Mat ROI_SD, ROI_mean;
	meanStdDev(a_timage_in, ROI_mean, ROI_SD);
	double reg;
	if (ROI_SD.at<double>(0,0) <= 1)
	{
		reg = 1;
	}
	else if (1<ROI_SD.at<double>(0,0) && ROI_SD.at<double>(0, 0) <= 1.4)
	{
		reg = sqrt(2);
	}
	else if (1.4<ROI_SD.at<double>(0,0) && ROI_SD.at<double>(0, 0) <= 2.8)
	{
		reg = 2 * sqrt(2);
	}
	else if (ROI_SD.at<double>(0,0)>2.8)
	{
		reg = 4 * sqrt(2);
	}
	return reg;
}

double Gabor_centeral_frequency(Mat a_timage_in)
{
	double SD = Gabor_standard_deviation(a_timage_in);
	if (SD==1)
	{
		return 0;
	}
	else if(SD==sqrt(2))
	{
		return 0.12;
	}
	else if(SD==2*sqrt(2))
	{
		return 0.8;
	}
	return 2;
}


double Gabor_GenvXY(Mat a_timage_in, double SD, int X, int Y, int X0, int Y0)
{
	return  1 / (2 * M_PI*pow(SD, 2))*exp(-1 * ((pow(X - X0, 2) + (pow(Y - Y0, 2))) / (2 * pow(SD, 2))));
}


int calculateOrientations(Mat a_timage_in, int theta_num)//回傳輸入圖片的最大角度
{
	Mat gradientX;
	Mat gradientY;
	Sobel(a_timage_in, gradientX, CV_32F, 1, 0, 3);
	Sobel(a_timage_in, gradientY, CV_32F, 0, 1, 3);
	// Create container element
	Mat orientation = Mat(gradientX.rows, gradientX.cols, CV_32F);
	
	// vector<int> theta_vector;
	int theta_vector[6];
	double Max_theta = 99999.9;
	int theta_output = 0;
	double theat_arry[6];
	for (int i = 0; i < theta_num; i++)
	{
		// theta_vector.push_back(180 / theta_num * i);
		theta_vector[i] = 180 / theta_num * i;
		theat_arry[i] = 0;
	}
	// Calculate orientations of gradients --> in degrees
	// Loop over all matrix values and calculate the accompagnied orientation
	normalize(gradientX, gradientX, 1, 0, NORM_L2);
	normalize(gradientY, gradientY, 1, 0, NORM_L2);
	for (int i = 0; i < gradientX.rows; i++) {
		for (int j = 0; j < gradientX.cols; j++) {
			// Retrieve a single value
			float valueX = gradientX.at<float>(i, j);
			float valueY = gradientY.at<float>(i, j);
			// Calculate the corresponding single direction, done by applying the arctangens function
			float result = fastAtan2(valueY, valueX);
			// Store in orientation matrix element
			orientation.at<float>(i, j) = result;
		}
	}
	//cout<<"orientation(rows, cols): "<<orientation.rows<<" , "<<orientation.cols<<endl;
	//cout<<orientation.at<float>(1, 30)<<endl;
	for (int i = 0; i < orientation.rows; i++)
	{
		for (int j = 0; j < orientation.cols; j++)
		{
			double diff_arry[6];
			if (orientation.at<float>(i, j) > 165+ numeric_limits<double>::epsilon())
			{
				orientation.at<float>(i, j) = orientation.at<float>(i, j) - 180;
			}
			for (int x = 0; x < theta_num; x++)
			{
				// cout<<"(i, j)-x:"<<i<<" "<<j<<" "<<x<<" :"<<orientation.at<float>(i, j)<<" - "<<theta_vector[x]<<endl; 
				//int a = orientation.at<float>(i, j);
				//int b = theta_vector[x];
				diff_arry[x] = abs(orientation.at<float>(i, j) - (float)theta_vector[x]);
				if (diff_arry[x] < Max_theta)
				{
					Max_theta= diff_arry[x];
					theta_output = theta_vector[x];
				}
			}
			double temple=orientation.at<float>(i, j) - theta_output;
			int position = theta_output / 30;
			if (i > 0 && i < orientation.rows - 1 && j>0 && j < orientation.cols - 1)
			{
				if (temple == 0.0)
				{
					theat_arry[int(position)]++;
				}
				else
				{
					if (temple >= 0)
					{
						theat_arry[position] += 1-(temple / 30);
					//	cout << 1 - (temple / 30) << endl;
						theat_arry[position + 1] += 1 - ((theta_output + 30 - orientation.at<float>(i, j)) / 30);
					//	cout << 1 - ((theta_output + 30 - orientation.at<float>(i, j)) / 30) << endl;
					}
					else
					{
						theat_arry[position] +=1-( abs(temple) / 30);
						//cout << 1 - (abs(temple) / 30) << endl;
						theat_arry[position - 1] += ((theta_output - orientation.at<float>(i, j)) / 30);
						//cout << ((theta_output-orientation.at<float>(i, j) ) / 30) << endl;
					}
				}
			}
			// cout<<"(i, j):"<<i<<" "<<j<<endl;
			Max_theta = 99999;
		}
	}

	int theta_counter = 0;
	for (int x = 0; x < theta_num; x++)
	{
		if (theat_arry[x] > theta_counter)
		{
			theta_output = x * 180/theta_num;
			theta_counter = theat_arry[x];
		}
	}
	return theta_output;
}


void Gabor_sub_region_parameter(Mat a_timage_in,int &main_orientation,double &SD_GR,double &center_frequency, int sub_region_counter, int theta_num)
{
	
	SD_GR = Gabor_standard_deviation(a_timage_in);
	center_frequency = Gabor_centeral_frequency(a_timage_in);
	main_orientation = calculateOrientations(a_timage_in, theta_num);
}





//int main(int argc, char const *argv[])
void feature_extract(const string& ROI_path)
{
	int A = 32; //子区域大小为32,32

	size_t len = ROI_path.size();

    vector<string> image;
	glob(ROI_path, image, false);

	
	for (size_t i = 0; i < image.size(); i++)
	{
		size_t sdf = image[i].find("Norm_ROI");
		string name;
		if(sdf != string::npos)
			name = image[i].substr(sdf + 9);
		else
            name = image[i];
		// Mat che = imread(save_imaginary + "norm_gabor_imaginary_" + name, 0);
		// if (che.empty() == 0)
		// {
		// 	waitKey(10);
		// }
		cout << image[i] << endl;
		Mat g_utimage_in = imread(image[i], 0);
		
		resize(g_utimage_in, g_utimage_in, Size(160, 160)); //resize为160,160
		//	int A = pow(2, Z+1);
		
		int sub_region_length = A;
			//sub_region_length = 32;
		if (sub_region_length > g_utimage_in.rows)
            return;
		Mat g_utimage_sub_region;
		Mat g_utimage_in_CLAHE;
		Mat g_utconvolution_real; //gabor滤波后的实部
		Mat g_utconvolution_imaginary; //gabor滤波后的虚部
		Mat g_utreal_horizontal;//實部水平累加
		Mat g_utreal_vertical;//實部垂直累加
		Mat g_utimaginary_horizontal;//虛部水平累加
		Mat g_utimaginary_vertical;//虛部垂直累加
		int sub_region_counter = g_utimage_in.rows / sub_region_length;//子区域计数，原图size应是子区域的整数倍
		double SD, center_frequency; //均值和方差
		int main_orientation; //主要均衡方向
		Mat sum;
		Mat sqsum;
		Mat g_utimage_POHE;

		//roi预处理
		Ptr<CLAHE> clahe = createCLAHE();
		clahe->setClipLimit(9); //cliplimit设为9.0
		clahe->apply(g_utimage_in, g_utimage_POHE); //局部自适应均衡化

		g_utimage_in_CLAHE = g_utimage_POHE.clone();
		Mat g_utsub_region_getedge;
		Mat reg;
		Mat g_utimage_parameter(sub_region_counter, sub_region_counter, CV_32FC3, Scalar(0));//存參數圖片，初始化为0.，size=(160/32,160/32, 3)=(15, 15, 3)，三个通道分别打算存储角度、均值和方差,double
		Mat g_utimage_theta(sub_region_counter, sub_region_counter, CV_8U, Scalar(0));//同g_utimage_parameter，只是儲存theta，size=(15, 15, 1)，uint8_t
		int theta_num = 6;//180度內取幾次做角度
		int c = 0, b = 0;
		
		for (int i = 0; i < sub_region_counter; i++)
		{
			for (int j = 0; j < sub_region_counter; j++)
			{
				reg = g_utimage_in_CLAHE(Rect(j*sub_region_length, i*sub_region_length, sub_region_length, sub_region_length));//這裡的ROI是指子區域(sub-region)
				g_utimage_sub_region = reg.clone();
				Gabor_sub_region_parameter(g_utimage_sub_region, main_orientation, SD, center_frequency, sub_region_counter, theta_num);
				g_utimage_parameter.at<Vec3f>(b, c)[0] = main_orientation;
				g_utimage_parameter.at<Vec3f>(b, c)[1] = SD;
				g_utimage_parameter.at<Vec3f>(b, c)[2] = center_frequency;
				
				Gabor_CR_real_imaginary_image(g_utimage_sub_region, g_utconvolution_real, g_utconvolution_imaginary, main_orientation, SD, center_frequency, sub_region_length);
				
				if (j == 0)
				{
					g_utreal_horizontal = g_utconvolution_real.clone();
					g_utimaginary_horizontal = g_utconvolution_imaginary.clone();
				}
				else
				{
					hconcat(g_utreal_horizontal, g_utconvolution_real, g_utreal_horizontal);
					hconcat(g_utimaginary_horizontal, g_utconvolution_imaginary, g_utimaginary_horizontal);
				}
				
				c++;
			}
			b++;
			c = 0;
			
			if (i == 0)
			{
				g_utreal_vertical = g_utreal_horizontal.clone();
				g_utimaginary_vertical = g_utimaginary_horizontal.clone();
			}
			else
			{
				vconcat(g_utreal_vertical, g_utreal_horizontal, g_utreal_vertical);
				vconcat(g_utimaginary_vertical, g_utimaginary_horizontal, g_utimaginary_vertical);
			}
		}
		
		medianBlur(g_utreal_vertical, g_utreal_vertical, 5);
		medianBlur(g_utimaginary_vertical, g_utimaginary_vertical, 5);
        imwrite(image[i], g_utreal_vertical);
        //imwrite(save_real + name, g_utreal_vertical);
        //imwrite(save_imaginary + "norm_gabor_imaginary_" + name, g_utimaginary_vertical);

    }
}

void Gabor_Kernel(Mat &a_tGabor_real,Mat &a_tGabor_imaginary,int Theta,double SD,double CF,int sub_region_length)//葛伯慮波核心產生器
{
	int gabor_size = sub_region_length;
	gabor_size = 5;
	int x, y;
	double xtmp, ytmp, tmp1, tmp2, tmp3;
	double re, im;
	//创建实部和虚部的gabor
	a_tGabor_real.create(gabor_size, gabor_size, CV_32F);
	a_tGabor_imaginary.create(gabor_size, gabor_size, CV_32F);
	double th = Theta*CV_PI / 180;
	for (int i = 0; i < gabor_size; i++) //定义模版大小  
	{
		for (int j = 0; j < gabor_size; j++)
		{
			x = j - gabor_size / 2;
			y = i - gabor_size / 2;

			xtmp = (double)x*cos(th) + (double)y*sin(th);
			ytmp = (double)y*cos(th) - (double)x*sin(th);

			tmp1 = exp(-(pow(xtmp, 2) + pow(ytmp, 2)) / (2 * pow(SD, 2)));
			tmp2 = cos(2 * CV_PI*xtmp*CF*CV_PI / 180);
			tmp3 = sin(2 * CV_PI*xtmp*CF*CV_PI / 180);

			re = tmp1*tmp2;
			im = tmp1*tmp3;

			a_tGabor_real.at<float>(i, j) = re;
			a_tGabor_imaginary.at<float>(i, j) = im;
			//printf("%f,%f\n",re,im);  
		}
	}
}

Mat Gabor_remove_DC( Mat a_timage_Gabor)
{
	Mat tmp_m, tmp_sd;
/*	double image_mean, SD;
	meanStdDev(a_timage_Gabor, tmp_m, tmp_sd);
	image_mean = tmp_m.at<double>(0, 0);
	SD = tmp_sd.at<double>(0, 0);*/
	double reg = 0;
	for (int i = 0; i < a_timage_Gabor.rows; i++)
	{
		for (int j = 0; j < a_timage_Gabor.cols; j++)
		{
			reg += a_timage_Gabor.at<float>(i, j);
		}
	}
	double image_mean = reg / pow(a_timage_Gabor.rows, 2);
	Mat a_timage_out(a_timage_Gabor.rows, a_timage_Gabor.cols, a_timage_Gabor.type(), Scalar(0));
	for (int i = 0; i < a_timage_Gabor.rows; i++)
	{
		for (int j = 0; j < a_timage_Gabor.cols; j++)
		{
			a_timage_out.at<float>(i, j) = a_timage_Gabor.at<float>(i, j) - image_mean;
		}
	}
	return a_timage_out;
}



void Gabor_CR_real_imaginary_image(Mat a_timage_in,Mat &a_tConv_real, Mat &a_tConv_imag,int theat, double SD, double CF, int sub_region_length)
{
	Mat a_tGabor_real;
	Mat a_tGaobr_imag;
	Gabor_Kernel(a_tGabor_real, a_tGaobr_imag, theat, SD, CF, sub_region_length);//每个子区域都要定义自适应的gabor滤波器
	//对filter做中心化（0均值化）
	a_tGabor_real = Gabor_remove_DC(a_tGabor_real);
	a_tGaobr_imag = Gabor_remove_DC(a_tGaobr_imag);
	filter2D(a_timage_in, a_tConv_real, CV_32F, a_tGabor_real);
	filter2D(a_timage_in, a_tConv_imag, CV_32F, a_tGaobr_imag);
	Mat a_real = a_tConv_real.clone();
	Mat a_imag = a_tConv_imag.clone();
	//二值图形逆转
	for (int i = 0; i < a_tConv_real.rows; i++)
	{
		for (int j = 0;j < a_tConv_real.cols; j++)
		{
			if (a_tConv_imag.at<float>(i, j) >=0)//>=0
			{
				a_tConv_imag.at<float>(i, j) = 255;
			}
			else
			{
				a_tConv_imag.at<float>(i, j) = 0;
			}
			if (a_tConv_real.at<float>(i, j)>=0)//>=0
			{
				a_tConv_real.at<float>(i, j) = 255;
			}
			else
			{
				a_tConv_real.at<float>(i, j) = 0;
			}
		}
	}

}
