#include <iostream>
#include <thread>

import logger;
import copier;

/// 解决方案:
/// 1. 截取图片的宽[1/4], 高[1/4] 区域, 识别该区域内的二维码
/// 2. 找到，左上角连续的 2x2 的二维码坐标，并计算获得完整标签的位置
/// 3. 对完整标签区域，获得标签与标签的行间距和列间距，以及左边距的信息
/// 4. 截图完整的左侧仅包含一列二维码的区域，获取对应二维码的位置
/// 5. 根据行间距以及每个完整标签的高度，截取出每个完整标签的每行标签区域，并保存成图片
int main()
{
    Logger::initialize();
    Logger::set_level(Logger::LogLevel::Debug);
    Copier copier;
    copier.copy();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout<<"输入任意字符并回车退出..."<<std::endl;
    std::cout << ">> ";
    std::string input;
    std::cin >> input;
    Logger::shutdown();
    return 0;
}