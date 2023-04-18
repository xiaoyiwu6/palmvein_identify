# palmvein_identify
掌静脉识别系统，实现了以下功能：
- 手势识别
- ROI提取
- 特征提取
- 特征识别
- GUI界面

# 项目配置需求（c++）
- opencv
- libtorch（pytorch支持的c++库）
- qt

# 源代码目录
-- palmvein_cpp\
&ensp;&ensp;-- gesture_roi        存放当前待识别特征的位置\
&ensp;&ensp;-- weights            存放神经网络权重参数的位置\
&ensp;&ensp;-- features           存放特征库的位置\
&ensp;&ensp;-- gui.cpp            
&ensp;&ensp;-- gabor_main.h/.cpp  特征提取算法，使用gabor自适应滤波器
&ensp;&ensp;-- load_torch.h/.cpp  神经网络加载\
&ensp;&ensp;-- utils.h/.cpp       实现录入、识别功能\
&ensp;&ensp;-- macro_ABI.h        部分嵌入版需要用到的宏\
&ensp;&ensp;-- UsbPalmCv.cpp      usb接口的摄像头驱动，可自己换源\
