# LaserImageCopy
镭射图片拷贝终端工具          

## 简介
LaserImageCopy 是一个用于将图片文件从一个目录拷贝到另一个目录的终端工具。    
固定文件夹路径，方便产线快速从采集电脑拷贝镭射提取图片至提取电脑。       

## 功能特点
- 支持多种图片格式（如 JPG、PNG、BMP 等）。
- 支持开启或关闭是否进行模型识别二维码截取图片功能。
- 支持日志记录，方便追踪拷贝过程中的问题。

## 文件说明
- main.cpp: 主程序文件。
- logger.cppm: 日志记录模块。
- detector.cppm: 模型识别模块。
- copier.cppm: 文件拷贝模块。
- config.json: 配置文件，包含源目录和目标目录路径等信息。
- README.md: 项目说明文件。
- CMakeLists.txt: CMake 构建配置文件。

## 项目编译依赖库
- C++20 标准库, 使用了模块特性。
- CMake 4.1 及以上版本。
- spdlog: 用于日志记录。
- OpenCV: 用于图像处理和模型识别。
- nlohmann/json: 用于处理 JSON 配置文件。
- TensorRT: 用于模型推理，项目使用版本 10.1.0.27。
- CUDA: 用于 GPU 加速，项目使用版本 12.5, cuda版本与TensorRT版本强相关。
- zbar: 用于二维码识别。
- indicators: 用于终端进度条显示。