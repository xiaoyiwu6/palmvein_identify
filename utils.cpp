 #include "utils.h"

std::vector<double> stdn{0.499053, 0.499097, 0.498960};
std::vector<double> meann{0.134352, 0.134359, 0.134314};

//save_path:roi保存文件名字
bool palmprint_detect_roi(torch::jit::script::Module &model, cv::Mat &img, const std::string &save_path, double threshold, bool toCUDA)
{
	std::cout<<"inputs: ["<<img.rows<<", "<<img.cols<<". "<<img.channels()<<"]"<<std::endl;
	cv::Mat img_bak = img.clone();
	int img_width = img.cols/2;//threshold_img.shape[0] -> threshold_img.rows 
	int img_height = img.rows/2;

	cv::cvtColor(img, img, cv::COLOR_BGR2RGB); //转为RGB
	cv::resize(img, img, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC); //缩放为二分之一
	cv::Mat img_;
	cv::resize(img, img_, cv::Size(256, 256), 0, 0, cv::INTER_CUBIC);
	img_.convertTo(img_, CV_32FC3, 1.0/255.0, -128.0/255.0); //输入预处理
	torch::Tensor img_tensor = torch::from_blob(img_.data, {1, img_.rows, img_.cols, 3}, torch::kFloat);//转为Tensor，NHWC
    if(toCUDA)
        img_tensor = img_tensor.permute({0, 3, 1, 2}).to(torch::kCUDA);//转为NCHW
    else
        img_tensor = img_tensor.permute({0, 3, 1, 2});//转为NCHW
	//std::cout<<img_tensor<<std::endl;

    torch::Tensor output = model.forward({img_tensor}).toTensor().cpu();
	//std::cout<<output<<std::endl;
	output = output.squeeze(0); //(1,42)->(42)
	//std::cout<<output.sizes()<<std::endl;

	//获取输出的坐标
	std::vector<cv::Point2f> contour;
	for(int i=0;i<21;i++)
		contour.emplace_back(output[2*i+0].item<float>()*img_width, output[2*i+1].item<float>()*img_height);
	std::vector<cv::Point2f> ac{contour[0], contour[2], contour[5], contour[9], contour[13], contour[17]};	
	//for(auto c:ac)
	//	std::cout<<"Point(x, y)"<<": "<<c.x<<" "<<c.y<<std::endl;

	// 计算轮廓的面积
	double area = cv::contourArea(ac);
	if(area<threshold)
		return false;
	//std::cout<<"area: "<<area<<std::endl;

	//计算要旋转的坐标
	float v2[2]{contour[5].x+contour[9].x, contour[5].y+contour[9].y};
	float v1[2]{contour[13].x+contour[17].x, contour[13].y+contour[17].y};
	//std::cout<<"v1, v2: "<<v1[0]<<" "<<v1[1]<<" "<<v2[0]<<" "<<v2[1]<<std::endl;

	if(v1[0]>v2[0])
		std::swap(v1, v2);
	//std::cout<<"v1, v2: "<<v1[0]<<" "<<v1[1]<<" "<<v2[0]<<" "<<v2[1]<<std::endl;
	double theta = atan2(v2[1]-v1[1], v2[0]-v1[0])*180./3.14159265358979323846;
	
	std::cout<<"The rotation of ROI is "<<theta<<std::endl;
	//旋转图片
	cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(v2[0], v2[1]), theta, 1.);
	cv::Mat img_r;
	R.convertTo(R, CV_32FC3);
	img_bak.convertTo(img_bak, CV_32FC3);
	cv::warpAffine(img_bak, img_r, R, cv::Size(img_width*2, img_height*2));
	std::cout<<"Rotatation complete."<<std::endl;
	
	//获取新坐标
	cv::Mat v1_mat = cv::Mat(2, 1, CV_32F, v1);
	v1_mat = cv::Mat((R(cv::Rect(0, 0, 2, 2)) * v1_mat + R.col(2))).reshape(1, 2).t();
	//std::cout<<"v1_mat: "<<format(v1_mat, cv::Formatter::FMT_PYTHON)<<std::endl;
	cv::Mat v2_mat = cv::Mat(2, 1, CV_32F, v2);
	v2_mat = cv::Mat(R(cv::Rect(0, 0, 2, 2)) * v2_mat + R.col(2)).reshape(1, 2).t();
	//std::cout<<"v2_mat: "<<format(v2_mat, cv::Formatter::FMT_PYTHON)<<std::endl;
	
	//std::cout<<"v1_mat: ["<<v1_mat.at<float>(0, 0)<<", "<<v1_mat.at<float>(0, 1)<<"]"<<std::endl;
	//std::cout<<"v2_mat: ["<<v2_mat.at<float>(0, 0)<<", "<<v2_mat.at<float>(0, 1)<<"]"<<std::endl;
	//获取roi坐标
	int ux = static_cast<int>(v1_mat.at<float>(0, 0)) - 40;
	int uy = static_cast<int>(v1_mat.at<float>(0, 1)) + 40;
	int lx = static_cast<int>(v2_mat.at<float>(0, 0)) + 40;
	int ly = static_cast<int>(v2_mat.at<float>(0, 1)) + 120 + static_cast<int>(v2_mat.at<float>(0, 0)) - static_cast<int>(v1_mat.at<float>(0, 0));
	//std::cout<<"ux, uy, lx, ly: "<<ux<<" "<<uy<<" "<<lx<<" "<<ly<<std::endl;

	//获取roi
	//std::cout<<img_r.size()<<std::endl;
	ux = std::max(0, ux);
	lx = std::min(479, lx);
	uy = std::max(0, uy);
	ly = std::min(639, ly);
	cv::Mat roi = img_r(cv::Rect(ux, uy, lx-ux, ly-uy));
	cv::resize(roi, roi, cv::Size(160, 160));

	//保存
	cv::imwrite(save_path, roi);
    std::cout<<save_path<<std::endl;
	
	return true;
}

void generate_all_features(const std::string &txt_path, const std::string &save_pt_path)
{
	auto model = torch::jit::load("/home/leo/projects/test_libtorch/pt/net.pt");
	std::ifstream file(txt_path);
	torch::Tensor features[10];
	int count = 0;
	for(int i=0;i<600;i++)
	{
		for(int j=0;j<10;j++)
		{
			std::string line;
			getline(file, line, '\t');
			//Todo: generate output from model net_tonji_preprocessed
			cv::Mat query_image = cv::imread("/home/leo/projects/treal/norm_gabor_real_"+line);	
			cv::cvtColor(query_image, query_image, cv::COLOR_BGR2RGB);
			cv::resize(query_image, query_image, cv::Size(128, 128));
			query_image.convertTo(query_image, CV_32FC3);
			// do the transforms
			cv::Mat dst;
			std::vector<cv::Mat> rgbChannels(3);
			cv::split(query_image, rgbChannels);
			for(int i=0;i<rgbChannels.size();i++)
			{
				rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0/255.0);
				rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0/stdn[i], (0-meann[i])/stdn[i]);
			}
			cv::merge(rgbChannels, dst);
			torch::Tensor img_tensor = torch::from_blob(dst.data, {1, dst.rows, dst.cols, 3}, torch::kFloat);//转为Tensor，NHWC
			img_tensor = img_tensor.permute({0, 3, 1, 2});
			std::cout<<img_tensor.sizes()<<std::endl;
			features[count++] = model.forward({img_tensor}).toTensor();
			
			getline(file, line);
		}
		//write to features named with label
		torch::Tensor features_cat = torch::cat(features, 0);
		std::cout<<features_cat.sizes()<<std::endl;
		torch::save({features_cat}, save_pt_path+std::to_string(i)+".pt");
		count = 0;

	}
}


void cat_all_features()
{
	std::vector<std::string> features_path;
	cv::glob("/home/leo/projects/palmvein_cplusplus_func/features", features_path, false);
	std::vector<torch::Tensor> features;
	torch::Tensor tmp;
	for(size_t i=0; i<features_path.size(); i++)
	{
		torch::load(tmp, features_path[i]);
		features.push_back(tmp);
		std::cout<<features_path[i]<<std::endl;
	}
	std::cout<<features.size()<<std::endl;
	auto features_cat = torch::cat(features, 0);
	torch::save(features_cat, "/home/leo/projects/palmvein_cplusplus_func/features/tonji_train_features.pt");
}

void generate_features(torch::jit::script::Module &model, const std::string &ROI_path, int label, bool toCUDA)
{
	std::vector<std::string> images; //用于记录ROI_PATH下的文件绝对路径名字
	cv::glob(ROI_path, images, false);	//记录ROI_PATH下的文件绝对路径名字
	std::vector<torch::Tensor> features(images.size()); //记录特征数组，用于cat合并
	for(size_t i=0; i<images.size(); i++)
	{
		cv::Mat img = cv::imread(images[i]);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		cv::resize(img, img, cv::Size(128, 128));
        img.convertTo(img, CV_32FC3);

		// do the transforms 做数据预处理
		cv::Mat dst;
		std::vector<cv::Mat> rgbChannels(3);
		cv::split(img, rgbChannels);
		for(int i=0;i<rgbChannels.size();i++)
		{
			rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0/255.0);
			rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0/stdn[i], (0-meann[i])/stdn[i]);
		}
		cv::merge(rgbChannels, dst);
		torch::Tensor img_tensor = torch::from_blob(dst.data, {1, dst.rows, dst.cols, 3}, torch::kFloat);//转为Tensor，NHWC
        if(toCUDA)
            img_tensor = img_tensor.permute({0, 3, 1, 2}).to(torch::kCUDA); // NCHW
        else
            img_tensor = img_tensor.permute({0, 3, 1, 2}); // NCHW
        features[i] = model.forward({img_tensor}).toTensor().cpu(); //喂入模型

	}
    torch::Tensor features_cat = torch::cat(features, 0); //合并
    torch::Tensor label_tensor = torch::full({images.size(), 1}, label);
    label_tensor = label_tensor.toType(torch::kFloat);
    features_cat = torch::cat({features_cat, label_tensor}, 1);
    std::string save_path = "../fatures/tonji_train_features.pt";
    torch::Tensor features_old_cat;
    torch::load(features_old_cat, save_path);
    torch::Tensor features_new_cat = torch::cat({features_old_cat, features_cat}, 0);
    torch::save({features_new_cat}, save_path);
}


int identify(torch::jit::script::Module &model, const std::string &img_name, const std::string &feature_path, double threshold, bool toCUDA)
{
	//read image and preprocess
	cv::Mat query_image = cv::imread(img_name);
	std::cout<<"read img success."<<std::endl;
	cv::cvtColor(query_image, query_image, cv::COLOR_BGR2RGB);
	cv::resize(query_image, query_image, cv::Size(128, 128));
	query_image.convertTo(query_image, CV_32FC3);
	//std::cout<<query_image.rows<<" "<<query_image.cols<<std::endl;

	// do the transforms
	cv::Mat dst;
	std::vector<cv::Mat> rgbChannels(3);
	cv::split(query_image, rgbChannels);
	for(int i=0;i<rgbChannels.size();i++)
	{
		rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0/255.0);
		rgbChannels[i].convertTo(rgbChannels[i], CV_32FC1, 1.0/stdn[i], (0-meann[i])/stdn[i]);
	}
    cv::merge(rgbChannels, dst);
	torch::Tensor img_tensor = torch::from_blob(dst.data, {1, dst.rows, dst.cols, 3}, torch::kFloat);//转为Tensor，NHWC

    //torch::Tensor img_tensor = torch::from_blob(query_image.data, {1, query_image.rows, query_image.cols, 3}, torch::kFloat);//转为Tensor，NHWC

    if(toCUDA)
        img_tensor = img_tensor.permute({0, 3, 1, 2}).to(torch::kCUDA);
    else
        img_tensor = img_tensor.permute({0, 3, 1, 2});
    std::cout<<img_tensor.sizes()<<std::endl;
    torch::Tensor output = model.forward({img_tensor}).toTensor().cpu();
	/*
	//加载所有特征矩阵名字
	std::vector<std::string> features;
	cv::glob(feature_path, features, false);
	if(features.size()==0)
		return -1;
	//加载所有特征和相应的标签
	std::vector<torch::Tensor> features_t;
	std::vector<int> labels;
	for(size_t i=0; i<features.size();i++)
	{
		//特征
		torch::Tensor tmp;
		torch::load(tmp, features[i]);
		features_t.emplace_back(tmp);
		//标签
		size_t pos = features[i].find(".pt");
		int label = std::stoi(features[i].substr(0, pos));
		//批量插入，使得labels的size和features_cat一样
		for(size_t j=0;j<tmp.size(0);j++)
		       labels.push_back(label);	
	}
	*/
	torch::Tensor features_cat;
	torch::load(features_cat, feature_path);
    int len = features_cat.size(1);
    torch::Tensor label_tensor = features_cat.index({"...", torch::indexing::Slice({len-1, torch::indexing::None})});
    label_tensor = label_tensor.squeeze();
    std::cout<<label_tensor.sizes()<<std::endl;
    features_cat = features_cat.index({"...",torch::indexing::Slice({0, len-1})});
    std::cout<<features_cat.sizes()<<std::endl;

	//欧氏距离匹配最近的特征
	torch::Tensor query_feature = output.reshape({1, 512}); //保证是(1, 512)
	torch::Tensor similarity_matrix = torch::cdist(query_feature.unsqueeze(0), features_cat.unsqueeze(0));
	torch::Tensor sorted, indices;
	std::tie(sorted, indices) = similarity_matrix.topk(10, -1, false);//取前十个
	indices = indices.squeeze();
	std::cout<<indices<<std::endl;
	similarity_matrix = similarity_matrix.squeeze();
	double sim = similarity_matrix[indices[0]].item<float>();
	//如果最近的距离大于阈值则返回0
	if(sim>threshold)
		return -1;
	//生成对应的标签
	std::unordered_map<int, int> label_map;
	std::cout<<"label candidate: ";
	for(int i=0; i<10;i++)
	{
        int label = label_tensor[indices[i]].item<int>();
	    std::cout<<label<<" ";
	    label_map[label]++;
	}
	std::cout<<std::endl;
	//找到最近距离的对应标签
	auto res = *max_element(label_map.begin(), label_map.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b)->bool{return a.second<b.second;});
	int match_label = res.first;
	//返回标签值
	return match_label;	

}
