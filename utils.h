#ifndef UTILS_H
#define UTILS_H

#include "macro_ABI.h"
#include "gabor_main.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <unistd.h>
#include <unordered_map>
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#define slots Q_SLOTS
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp" //OpenCV highgui模块头文件
#include "opencv2/imgproc/imgproc.hpp" //OpenCV 图像处理头文件
#include <opencv2/core/hal/interface.h>

bool palmprint_detect_roi(torch::jit::script::Module &model, cv::Mat &img, const std::string &save_path, double threshold, bool toCUDA);

int identify(torch::jit::script::Module &model, const std::string &img_name, const std::string &feature_path, double threshold, bool toCUDA);

void generate_features(torch::jit::script::Module &model, const std::string &img_path, int label, bool toCUDA); //根据roi路径和标签生成特征，特征为.pt文件，文件名即为标签

void generate_all_features(const std::string &txt_path, const std::string &save_pt_path); //生成所有的特征，用不到

//int identify(torch::jit::script::Module &model, const std::string &img_name, double threshold=3.);//识别，特征匹配方法

extern std::vector<double> stdn;
extern std::vector<double> meann;

#endif
