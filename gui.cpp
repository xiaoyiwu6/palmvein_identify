#include <QWidget>
#include <QImage>
#include <QTimer>
#include <QLabel>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPixmap>
#include <QMessageBox>
#include <QApplication>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp" //OpenCV highgui模块头文件
#include "opencv2/imgproc/imgproc.hpp" //OpenCV 图像处理头文件
#include "opencv2/core/types.hpp"
#include "usb_dev.h"
#include <fstream>
#include <stdlib.h>
#include "load_torch.h"
#include "utils.h"
#include <cstdlib>
#include <pthread.h>
#include <time.h>

torch::jit::script::Module rexnet, net;

// global variable to control camera

//QImage *matToimage(Mat& mat)
//{
//    if(mat.type() != CV_8UC1) {
//        return nullptr;
//    }
//    QImage *img = new QImage(mat.cols, mat.rows,QImage::Format_Grayscale8);
//    uchar *prtImage = img->bits();
//    uchar *prtMat = mat.data;
//    memcpy(prtImage, prtMat, mat.cols* mat.rows);
//    return img;
//}

class MainWindow : public QWidget {
public:
    MainWindow() {
        // 创建左侧的空白区域
        image_label = new QLabel(this);
        image_label->setAlignment(Qt::AlignCenter);
        image_label->setStyleSheet("border: 1px solid black;");
        image_label->setFixedSize(480, 640);

        // 创建右侧的三个按钮
        button1 = new QPushButton("录入", this);
        button1->setFixedHeight(100);
        button1->setFixedWidth(300);
        button2 = new QPushButton("检测", this);
        button2->setFixedHeight(100);
        button2->setFixedWidth(300);
        button3 = new QPushButton("识别", this);
        button3->setFixedHeight(100);
        button3->setFixedWidth(300);

        // 为三个按钮分别绑定响应函数
        connect(button1, &QPushButton::clicked, this, &MainWindow::add_feature);
        connect(button2, &QPushButton::clicked, this, &MainWindow::detect);
        connect(button3, &QPushButton::clicked, this, &MainWindow::predict);

        // 创建一个垂直布局，并添加左右两个部件
        layout = new QHBoxLayout(this);
        layout->addWidget(image_label, 4);
        button_layout = new QVBoxLayout();
        button_layout->addWidget(button1);
        button_layout->addWidget(button2);
        button_layout->addWidget(button3);
        layout->addLayout(button_layout, 1);

        // 设置窗口的标题和大小
        setWindowTitle("掌静脉识别系统");
        setGeometry(100, 100, 800, 600);

        // open camera
        system("sudo pkill -9 UsbPalmCv");
        system("sudo ../UsbPalmCv &");
    }

private:
    QLabel* image_label;
    QPushButton* button1;
    QPushButton* button2;
    QPushButton* button3;
    QHBoxLayout* layout;
    QVBoxLayout* button_layout;

    void add_feature() {
        // 处理录入功能
        //QMessageBox::warning(this,"Title","Error Message");

        ifstream file;
        file.open("../features/membernum.txt");
        if(!file.is_open()) {
            printf("open file failed.");
            return;
        }
        int num;
        file >> num;
        file.close();
        printf("There are %d members in features for now.\n", num);
        num++;

        camera_thread(15);

        //feature_extract("../gesture_roi", "../real", "../imaginary");
        QMessageBox::information(this,  "提示",  "左手录入成功!");
        camera_thread(15);
        QMessageBox::information(this,  "提示",  "右手录入成功!");

        generate_features(net, "../gesture_roi", num, false);
        ofstream out;
        out.open("../features/membernum.txt");
        out.flush();
        out << num;
        out.close();
    }

    void detect() {
        // 处理检测功能
        int num = 5;
        camera_thread(num);
        //roi_extration();
        QMessageBox::information(this,  "提示",  "检测成功!");

    }

    void predict() {
        // 处理识别功能
        vector<string> image;
        glob("../gesture_roi", image, false);
        unordered_map<int, int> label_map;//record the most predict label
        for(int i = 0; i < image.size(); i++) {
            int temp = identify(net, image[i], "/home/leo/projects/palmvein-cpp/features/tonji_train.pt", 8.0, false);
            if(temp!=-1)
                label_map[temp]++;
        }
        int res = -1;
        if(!label_map.empty())
            res = (*max_element(label_map.begin(), label_map.end(), [](const pair<int, int> &a, const pair<int, int> &b)->bool{return a.second<b.second;})).first;

        QMessageBox::information(this,  "提示",  QString::fromStdString("识别成功! 人员编号为: "+ to_string(res)));

        //?
        /*//////////////////////////////////*/
        //system("sudo rm ../features/*");

    }

    /*///////////////////////////*/
    // try to reduce loading time
    void camera_thread(int num) {
        // system("pwd");

        if((access("myfifo", 0) == -1)){
            int ret = mkfifo("myfifo", 0777);
            if (ret == -1)
            {
                printf("Make fifo error\n");
                return;
            }
        }

        int32_t fd = open("myfifo", O_RDONLY|O_NONBLOCK);
        int32_t length = 307205;
        uint8_t Datas[length] = {0};
        int count = 0;
        std::string path = "../gesture_roi/" + to_string(count) + ".jpg";

        while(count < num) {
            read(fd, Datas, length);
            cv::Mat image;
            toMat(Datas, image);
            if(image.rows != 640 || image.cols != 480)
                continue;
            /*
            QImage qimg(image.cols, image.rows, QImage::Format_Indexed8);
            qimg.setColorCount(256);
            for(int i = 0; i < 256; i++) {
                qimg.setColor(i, qRgb(i, i, i));
            }
            */
            clock_t start,end;
            //start = clock();
            // multithread start at begining

            //end = clock();
            //cout<<double(end-start)/1000000<<endl;
            cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
            cv::imshow("image", image);
            waitKey(30);
            if(palmprint_detect_roi(rexnet, image, path, 15000, true))
                path = "../gesture_roi/" + to_string(++count) + ".jpg";
        }
        cv::destroyAllWindows();

        feature_extract("../gesture_roi/");
    }

};
// thread to evaluate model rexnet
void *evaluate_thread1(void *arg){
    // random picture
    torch::Tensor img = torch::rand({1,256,256,3});
    img = img.permute({0, 3, 1, 2}).to(torch::kCUDA);
    torch::Tensor output = rexnet.forward({img}).toTensor();
}

// thread1 to load and evaluate the model rexnet
void *preload_thread1(void *arg){
    load_module("../weights/rexnet.pt", rexnet, true);

    // useless to warmup model
    /*
    pthread_t t1, t2, t3, t4;
    int err;
    err = pthread_create(&t1, NULL, evaluate_thread1, NULL);
    if(err != 0) printf("load model rexnet failed!");
    err = pthread_create(&t2, NULL, evaluate_thread1, NULL);
    if(err != 0) printf("load model rexnet failed!");
    err = pthread_create(&t3, NULL, evaluate_thread1, NULL);
    if(err != 0) printf("load model rexnet failed!");
    err = pthread_create(&t4, NULL, evaluate_thread1, NULL);
    if(err != 0) printf("load model rexnet failed!");
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);*/
    // system("sudo rm ../gesture_roi/*");
}

// thread2 to load and evaluate the model net
void *preload_thread2(void *arg){
    load_module("../weights/net.pt", net, false);
    // warmup model
    /*torch::Tensor img = torch::rand({1,128,128,3});
    img = img.permute({0, 3, 1, 2}).to(torch::kCUDA);
    net.forward({img});*/
}

int main(int argc, char* argv[])
{
    pthread_t thread1, thread2;
    int err;
    // multithread preload
    err = pthread_create(&thread1, NULL, preload_thread1, NULL);
    if(err != 0) printf("load model rexnet failed!");
    err = pthread_create(&thread2, NULL, preload_thread2, NULL);
    if(err != 0) printf("load model net failed!");

    QApplication a(argc, argv);
    MainWindow w;
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    w.show();


    return a.exec();


}
