#include <unistd.h>
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp" //OpenCV highgui模块头文件
#include "opencv2/imgproc/imgproc.hpp" //OpenCV 图像处理头文件
#include "usb_dev.h"

using namespace std;
using namespace cv;

#define row 640  
#define col 480 

#define RX_BUF_SIZE	512

typedef enum {
    PIXFORMAT_RGB565,    // 2BPP/RGB565
    PIXFORMAT_YUV422,    // 2BPP/YUV422
    PIXFORMAT_GRAYSCALE, // 1BPP/GRAYSCALE
    PIXFORMAT_JPEG,      // JPEG/COMPRESSED
    PIXFORMAT_RGB888,    // 3BPP/RGB888
    PIXFORMAT_RAW,       // RAW
    PIXFORMAT_RGB444,    // 3BP2P/RGB444
    PIXFORMAT_RGB555,    // 3BP2P/RGB555
} pixformat_t;

static bool LightAutoAdj = true;
static uint16_t Light850_Value = 6;
static uint16_t Light940_Value = 0;
static int16_t expValue = 0;
static int8_t contrastLevel = 0;

sem_t sem[5];

void *GetImageThread( void *arg );

//
//	ImageShow().
//
void ImageShow(char* showNamed,uint8_t *Datas)
{
	uint16_t width = (Datas[0] << 8) | Datas[1];
	uint16_t height = (Datas[2] << 8) | Datas[3];
	uint8_t format = Datas[4];

	uint8_t *SrcPtr = &Datas[5];

	Mat imageGray;

	Mat image;

	if( format == PIXFORMAT_GRAYSCALE )
	{
		image = Mat( height, width, CV_8UC1, SrcPtr );

		imageGray = image;
	}
	else if( format == PIXFORMAT_YUV422 )
	{
		Mat imageSrc = Mat( height, width, CV_8UC2, SrcPtr );

		image = Mat( height, width, CV_8UC3 );

		cvtColor( imageSrc,image, 115 );

		cvtColor( image,imageGray, 7 );
	}

//	Mat kern = (Mat_<char>(3,3) << 0, -1 ,0,
//                               -1, 5, -1,
//                               0, -1, 0);
//	filter2D(image_roi, imageDst, image_roi.depth(), kern );

//	Mat image1 = image;
//	GammaCorrection( image1, image2, gamma );

	Mat imageRotate = Mat::zeros(imageGray.size(), imageGray.type());

	transpose(imageGray, imageRotate);
	flip(imageRotate, imageRotate, -1);

	Mat imageShow;

	imageShow = imageRotate;

	resizeWindow(showNamed, imageShow.cols, imageShow.rows );


	imshow(showNamed, imageShow );	
	/*user programer*/
	
}
//
//	DataHandle().
//
void DataHandle( uint8_t *Datas )
{
	ImageShow("show",Datas);
	waitKey(30);
	free(Datas);
}
//
//	main().
//
/*
int main(int argc, char *agrv[])
{
	pthread_t ntid;
	int err;

    if( usb_dev_init() == false )
    {
        printf("usb_dev_init fail!\n");
	
		return -1;
    }

    printf("usb_dev_init ok\n");

    if( usb_dev_open( 0x303A, 0x3002 ) == false )
    {
		printf("usb_dev_open fail!\n");

		return -1;
    }

    printf("usb_dev_open ok\n");

	err = pthread_create(&ntid, NULL,GetImageThread, NULL );

	if( err != 0 )
	{
		printf("create get Image thread is failed\n");		
		return -1;
	}

	sem_init(&sem[0], 0, 1);

	Mat dstimg(row, col, CV_8UC3, Scalar(0, 0, 0));
 	namedWindow("show", WINDOW_AUTOSIZE);
	imshow("show", dstimg);

	while(1)
	{
		sleep(10);
	}
    
	printf("app exit\n");
 

    return 0;
}*/
//
//	main().
//
void *GetImageThread( void *arg )
{
	uint8_t DataBuf[1024*1024];
	uint8_t CapCnt = 0;
	uint32_t RecvCnt = 0;
	uint32_t PacketLen = 0;
	uint8_t HeadFlag = 0;

	uint8_t TxBuff[64];
	uint8_t RxBuff[RX_BUF_SIZE];

	while( usb_dev_isopen() )   
	{
		if( (HeadFlag == 0) && (RecvCnt == 0 ) )
		{
			if (LightAutoAdj == true)
			{
				if (CapCnt == 3)
				{
					Light850_Value = 0;
					Light940_Value = 6;
				}
				else if (CapCnt == 6)
				{
					Light850_Value = 4;
					Light940_Value = 4;
				}
				else if (CapCnt >= 9)
				{
					CapCnt = 0;

					Light850_Value = 6;
					Light940_Value = 0;
				}

			}			

			TxBuff[0] = 0xAA;
			TxBuff[1] = 0x05;
			TxBuff[2] = Light850_Value;				//LightValue1.
			TxBuff[3] = Light940_Value;				//LightValue2.
			TxBuff[4] = (expValue >> 8) & 0xFF;						//Exposure time added value.
			TxBuff[5] = expValue & 0xFF;
			TxBuff[6] = contrastLevel;
			TxBuff[7] = 0x55;

			if( usb_dev_write_sync( (uint8_t *)TxBuff,8, 1000 ) == false )
			{
				break;
			}
			
//			printf("usb_dev_write_sync\n");
		}

		int ret = usb_dev_read_sync( RxBuff, RX_BUF_SIZE, 5000 );

		if( ( ret < 0 ) )
		{
			printf("usb_dev_read_sync fail,%d\n", ret);
			break;
		}

		memcpy( &DataBuf[RecvCnt], RxBuff, ret );

		RecvCnt += ret;

		if( HeadFlag != 0 )
		{
			printf("HeadFlag!=0\n");
			if( RecvCnt >= PacketLen )
			{
				if( DataBuf[PacketLen - 1] == 0xAA )
				{
					uint8_t *Datas = (uint8_t *)malloc( PacketLen - 7 );
					if( Datas != NULL )
					{
						printf("get Datas\n");
						memcpy(Datas, &DataBuf[6], PacketLen - 7 );

						ImageShow("show",Datas);
						waitKey(30);
						
						free(Datas);
						
					}
				}
				else
				{
					printf("Packet error\n");
				}

				HeadFlag = 0;
				RecvCnt = 0;
			}
		}
		else if( RecvCnt >= 6 )
		{
			for( uint32_t i = 0; i < RecvCnt; i ++ )
			{
				if( DataBuf[i] == 0xA5 )
				{
					if( DataBuf[i + 1] == 0x5A )
					{
						PacketLen = (DataBuf[i + 2] << 24) | (DataBuf[i + 3] << 16) | (DataBuf[i + 4] << 8) | DataBuf[i + 5];
						PacketLen += 3;
				
						if( PacketLen < 1024*1024 )
						{
							memcpy( DataBuf, &DataBuf[i], RecvCnt - i );
							RecvCnt = RecvCnt - i;

							HeadFlag = 1;
						}						
					}
				}
			}
		}
		else if( RecvCnt >= 1024*1024 )
		{
			RecvCnt = 0;
			printf("RecvCnt Overlength\n");
		}
		
	}

    usb_dev_close();
    usb_dev_exit();

	return NULL;
}
