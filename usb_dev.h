#ifndef _USB_DEV_H_
#define _USB_DEV_H_

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <errno.h>
#include <fcntl.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp" //OpenCV highgui模块头文件
#include "opencv2/imgproc/imgproc.hpp" //OpenCV 图像处理头文件
#include "libusb-1.0/libusb.h"
#include "utils.h"


bool usb_dev_isopen( void );
bool usb_dev_init( void );
void usb_dev_exit( void );
bool usb_dev_open( uint16_t vendor_id, uint16_t product_id );
void usb_dev_close(void);
bool usb_dev_write_sync( uint8_t *Datas, uint16_t DataLen, int timeout );
int usb_dev_read_sync( uint8_t *Buf, uint16_t bufsz, int timeout );

// void ImageShow(std::string showNamed, uint8_t *Datas);
void ImageShow(uint8_t *Datas);
void* GetImageThread(void *);
void toMat(uint8_t *Datas, Mat &dst);
//void camera_thread(int num);

#endif
