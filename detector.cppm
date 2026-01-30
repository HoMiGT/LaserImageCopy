//
// Created by houmin on 2026/1/16.
//
module;

#include <algorithm>
#include <array>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <zbar.h>
#include <opencv2/opencv.hpp>

export module detecter;


export enum class QrOrientation {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3,
	UNKNOW = -1
};

export inline std::ostream& operator<<(std::ostream& os, const QrOrientation& orientation)
{
    switch (orientation)
    {
    case QrOrientation::UP:
        os << "UP";
        break;
    case QrOrientation::RIGHT:
        os << "RIGHT";
        break;
    case QrOrientation::DOWN:
        os << "DOWN";
        break;
    case QrOrientation::LEFT:
        os << "LEFT";
        break;
    case QrOrientation::UNKNOW:
        os << "UNKNOW";
        break;
    default:
        os << "INVALID";
        break;
    }
    return os;
}

export struct QrCodeResult {
    std::string context;
    QrOrientation orientation;
    cv::Point2i LeftTop;
    cv::Point2i LeftBottom;
    cv::Point2i RightBottom;
    cv::Point2i RightTop;
};

/**
 * @brief Zbar 二维码识别类
 */
export class Recognize
{
public:
    Recognize();
    ~Recognize()=default;

    /**
     * @brief 检测二维码
     * @param gray 单通道灰度图像
     * @param qrcode 识别到的二维码内容
     * @param orientation 二维码方向 0:上 1:右 2:下 3:左
     * @return
     */
    bool detect(const cv::Mat &input_mat, QrCodeResult& qrret);
    [[nodiscard]] std::string what() const;

private:
    zbar::ImageScanner m_scanner;
    std::string m_error;
};

Recognize::Recognize()
{
    m_scanner.set_config(zbar::ZBAR_QRCODE,zbar::ZBAR_CFG_ENABLE,1);
}

bool Recognize::detect(const cv::Mat &input_mat, QrCodeResult& qrret)
{
    cv::Mat gray;
    if (input_mat.channels() !=1)
    {
		cv::cvtColor(input_mat, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input_mat;
    }
    const auto width = gray.cols;
    const auto height = gray.rows;
    zbar::Image img(width,height,"Y800",gray.data,width*height);
    bool is_ok{ false };
    try
    {
        if (const auto ret = m_scanner.scan(img); ret >0 )
        {
            auto high_quality = 0;
            for (auto symbol = img.symbol_begin(); symbol != img.symbol_end(); ++symbol)
            {
                if (const auto quality = symbol->get_quality(); quality > high_quality)
                {
					qrret.context = symbol->get_data();
                    const auto orientation = symbol->get_orientation();
                    if (orientation == 0)
                    {
                        qrret.orientation = QrOrientation::UP;
                    }
                    else if (orientation == 1)
                    {
                        qrret.orientation = QrOrientation::RIGHT;
                    }
                    else if (orientation == 2)
                    {
                        qrret.orientation = QrOrientation::DOWN;
                    }
                    else if (orientation == 3)
                    {
                        qrret.orientation = QrOrientation::LEFT;
                    }
                    else {
						qrret.orientation = QrOrientation::UNKNOW;
                    }
					qrret.LeftTop = cv::Point2i(symbol->get_location_x(0), symbol->get_location_y(0));
					qrret.LeftBottom = cv::Point2i(symbol->get_location_x(1), symbol->get_location_y(1));
					qrret.RightBottom = cv::Point2i(symbol->get_location_x(2), symbol->get_location_y(2));
					qrret.RightTop = cv::Point2i(symbol->get_location_x(3), symbol->get_location_y(3));
                    high_quality = quality;
                    is_ok = true;
                }else
                {
                    img.set_data(nullptr,0);
                    return false;
                }
            }
        }
        img.set_data(nullptr,0);
        return is_ok;
    }catch (std::exception& e)
    {
        m_error = e.what();
        img.set_data(nullptr,0);
        return false;
    }
}

std::string Recognize::what() const
{
    return m_error;
}

class NvLogger: public nvinfer1::ILogger
{
    void log(const nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kERROR)
        {
            std::cerr << "NvLogger error: " << msg << std::endl;
        }
    }
};

struct TrtDestroy
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
            delete obj;
    }
};

template<typename T>
using unp = std::unique_ptr<T,TrtDestroy>;

/**
 * @brief TensorRT 位置检测类
 */
export class Location
{
public:
    explicit Location(std::string_view model_path);
    ~Location();

    bool build();
    bool infer(const cv::Mat& src, std::vector<cv::Rect2i>& boxes);
    [[nodiscard]] std::string what() const;

private:
    std::string m_model_path{};
    unp<nvinfer1::IRuntime> m_runtime{nullptr};
    unp<nvinfer1::ICudaEngine> m_engine{nullptr};
    unp<nvinfer1::IExecutionContext> m_context{nullptr};
    int m_input_index{0};
    int m_output_index{1};
    const char* m_input_name{nullptr};
    const char* m_output_name{nullptr};
    nvinfer1::Dims m_input_dims{};
    nvinfer1::Dims m_output_dims{};
    size_t m_input_size{0};
    size_t m_output_size{0};
    std::string m_error{};
    std::array<void*,2> m_buffers{};
    std::vector<__half> m_input_data;
    std::vector<__half> m_output_data;
    std::vector<float> m_output_float_data;
    cudaStream_t m_stream{};
    double m_scale{ 0.0 };
    int m_width{0};
    int m_height{0};
	int m_letter_width{ 0 };
	int m_letter_height{ 0 };

    void letterbox(const cv::Mat& src, cv::Mat& dst, size_t nw, size_t nh);
    bool preprocess(const cv::Mat& src) noexcept;
    bool postprocess(std::vector<cv::Rect2i>& boxes) noexcept;
    cv::Rect2i de_letterbox(const cv::Rect2i&& box) noexcept;
    bool serialize();

    static size_t get_size_by_dims(const nvinfer1::Dims& dims);
    static void print_tensor_info(const nvinfer1::ICudaEngine* engine);
    static void destroy_builder(const nvinfer1::IBuilder* builder, const nvinfer1::IBuilderConfig* config, const nvinfer1::INetworkDefinition* network);
};

Location::Location(const std::string_view model_path)
    :m_model_path(model_path)
{
}

Location::~Location()
{
    if (m_buffers[m_input_index])
    {
        cudaFree(m_buffers[m_input_index]);
    }
    if (m_buffers[m_output_index])
    {
        cudaFree(m_buffers[m_output_index]);
    }
    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
    }
}

bool Location::build()
{
    if (!serialize())
    {
        m_error = std::format("Model not found, {}", m_model_path);
        return false;
    }
    std::ifstream engine_file(m_model_path, std::ifstream::binary);
    if (!engine_file)
    {
        m_error = std::format("Failed to open engine file.");
        return false;
    }
    engine_file.seekg(0, std::ifstream::end);
    const auto engine_size = engine_file.tellg();
    engine_file.seekg(0, std::ifstream::beg);
    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(),engine_size);
    engine_file.close();
    NvLogger logger;
    m_runtime = unp<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    m_engine = unp<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_data.data(),engine_size));
    print_tensor_info(m_engine.get());
    m_context = unp<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    m_input_name = m_engine->getIOTensorName(m_input_index);
    m_output_name = m_engine->getIOTensorName(m_output_index);
    m_input_dims = m_engine->getTensorShape(m_input_name);
    m_output_dims = m_engine->getTensorShape(m_output_name);
    m_input_size = get_size_by_dims(m_input_dims);
    m_output_size = get_size_by_dims(m_output_dims);
    m_input_data.resize(m_input_size);
    m_output_data.resize(m_output_size);
    m_output_float_data.resize(m_output_size);
    cudaMalloc(&m_buffers[m_input_index], sizeof(__half)* m_input_size);
    if (!m_context->setInputTensorAddress(m_input_name, m_buffers[m_input_index]))
    {
        m_error = "Failed to set input tensor address.";
        if (m_buffers[m_input_index])
        {
            cudaFree(m_buffers[m_input_index]);
        }
        return false;
    }
    cudaMalloc(&m_buffers[m_output_index], sizeof(__half) * m_output_size);
    if (!m_context->setOutputTensorAddress(m_output_name, m_buffers[m_output_index]))
    {
        m_error = "Failed to set output tensor address.";
        if (m_buffers[m_input_index])
        {
            cudaFree(m_buffers[m_input_index]);
        }
        if (m_buffers[m_output_index])
        {
            cudaFree(m_buffers[m_output_index]);
        }
        return false;
    }

    if (cudaStreamCreate(&m_stream) != cudaSuccess)
    {
        m_error = "Failed to create CUDA stream.";
        if (m_buffers[m_input_index])
        {
            cudaFree(m_buffers[m_input_index]);
        }
        if (m_buffers[m_output_index])
        {
            cudaFree(m_buffers[m_output_index]);
        }
        return false;
    }
    return true;
}

bool Location::infer(const cv::Mat& src, std::vector<cv::Rect2i>& boxes)
{
    if (!preprocess(src))
    {
        return false;
    }
    if (const auto ret = cudaMemcpyAsync(m_buffers[m_input_index],m_input_data.data(),
        sizeof(__half)* m_input_size,cudaMemcpyHostToDevice, m_stream); ret!=cudaSuccess)
    {
        m_error = std::format("CUDA asynchronous data synchronization failed: {}", cudaGetErrorString(ret));
        return false;
    }
    try
    {
        if (!m_context->enqueueV3(m_stream))
        {
            return false;
        }
    }catch (std::exception& e)
    {
        m_error = std::format("Failed to enqueueV3, {}", e.what());
        return false;
    }
    if (const auto ret = cudaMemcpyAsync(m_output_data.data(),m_buffers[m_output_index],
        sizeof(__half)*m_output_size,cudaMemcpyDeviceToHost,m_stream); ret!=cudaSuccess)
    {
        m_error = std::format("CUDA asynchronous data synchronization failed: {}", cudaGetErrorString(ret));
        return false;
    }
    if (const auto ret = cudaStreamSynchronize(m_stream); ret!=cudaSuccess)
    {
        m_error = std::format("CUDA Stream synchronization failed: {}", cudaGetErrorString(ret));
        return false;
    }
    if (!postprocess(boxes))
    {
		m_error = std::format("Postprocess failed: {}", m_error);
        return false;
    }
    return true;
}

std::string Location::what() const
{
    return m_error;
}

void Location::letterbox(const cv::Mat& src, cv::Mat& dst, const size_t nw, const size_t nh)
{
    m_width = src.cols;
    m_height = src.rows;
    m_letter_width = nw;
    m_letter_height = nh;
    const auto w = static_cast<double>(m_width);
    const auto h = static_cast<double>(m_height);
    const auto nwd = static_cast<double>(nw);
    const auto nhd = static_cast<double>(nh);
    m_scale = std::min(nhd / h, nwd / w);
    const auto sw = std::round(m_scale * w);
    const auto sh = std::round(m_scale * h);
    auto dw = nwd - sw;
    auto dh = nhd - sh;
    dw /= 2.0;
    dh /= 2.0;
    if (w!=nwd || h != nhd)
    {
        cv::resize(src,dst,cv::Size(static_cast<int>(sw),static_cast<int>(sh)), m_scale);
    }else
    {
        dst = src;
    }
    const auto top = static_cast<int>(std::round(dh-0.1));
    const auto bottom = static_cast<int>(std::round(dh+0.1));
    const auto left = static_cast<int>(std::round(dw-0.1));
    const auto right = static_cast<int>(std::round(dw+0.1));
    cv::copyMakeBorder(dst,dst,top,bottom,left,right,cv::BORDER_CONSTANT, cv::Scalar{114,114,114});
}

bool Location::preprocess(const cv::Mat& src) noexcept
{
    try
    {
        const auto channels = m_input_dims.d[1];
        const auto height = m_input_dims.d[2];
        const auto width = m_input_dims.d[3];
        cv::Mat dst;
        letterbox(src, dst, width,height);
        //m_letter_dst = dst.clone();
        dst.convertTo(dst, CV_32F, 1.0f/ 255.0f);
        cvtColor(dst, dst, cv::COLOR_BGR2RGB);
        std::vector<cv::Mat> rgb;
        rgb.reserve(channels);
        for (auto i{0}; i < channels; ++i)
        {
            rgb.emplace_back(cv::Mat{static_cast<int>(width),static_cast<int>(height),CV_32FC1});
        }
        cv::split(dst,rgb);
        for (auto i{0};i <channels; ++i)
        {
            const auto& channel = rgb[i];
            const auto size = channel.total();
            const auto data = reinterpret_cast<float*>(channel.data);
            for (auto j{0}; j < size; ++j)
            {
                m_input_data[i*size +j] = __float2half(data[j]);
            }
        }
        return true;
    }catch (std::exception& e)
    {
        m_error = std::format("preprocess error: {}", e.what());
        return false;
    }
}

bool Location::postprocess(std::vector<cv::Rect2i>& boxes) noexcept
{
    try
    {
        constexpr auto score_threshold = 0.5f;
        constexpr auto nms_threshold = 0.5f;
        const auto classes = m_output_dims.d[1];
        const auto confidence = m_output_dims.d[2];
        std::ranges::transform(m_output_data.begin(),m_output_data.end(),m_output_float_data.begin(),
            [](const __half a){return __half2float(a);});
        cv::Mat output_mat{
            static_cast<int>(classes),
            static_cast<int>(confidence),
            CV_32F,
            m_output_float_data.data()
        };
        cv::transpose(output_mat,output_mat);
        const auto rows = output_mat.rows;
        const auto cols = output_mat.cols;
        std::vector<int> class_ids;
        std::vector<float> class_scores;
        cv::Point class_id;
        std::vector<cv::Rect2i> suspected_boxes;
        double max_class_score{0.0};
        for (auto r{0}; r<rows; ++r)
        {
            cv::Mat scores = output_mat.row(r).colRange(4, cols);
            cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id);
            if (max_class_score < score_threshold)
            {
                continue;
            }
            class_scores.emplace_back(static_cast<float>(max_class_score));
            const auto cx = output_mat.at<float>(r,0);
            const auto cy = output_mat.at<float>(r,1);
            const auto w = output_mat.at<float>(r,2);
            const auto h = output_mat.at<float>(r,3);
            const auto left = static_cast<int>(cx - 0.5 * w);
            const auto top = static_cast<int>(cy - 0.5 * h);
            suspected_boxes.emplace_back(de_letterbox(cv::Rect2i{left,top,static_cast<int>(w),static_cast<int>(h)}));
        }
        cv::dnn::NMSBoxes(suspected_boxes, class_scores, score_threshold, nms_threshold,class_ids);
        for (const int id : class_ids)
        {
			const auto& box = suspected_boxes[id];
            const auto aspect_ratio = static_cast<float>(box.width) / static_cast<float>(box.height);
            // 只保留接近正方形的框
            if (aspect_ratio < 0.8f || aspect_ratio > 1.2f)
            {
                continue;
            }
            boxes.emplace_back(suspected_boxes[id]);
        }
        return true;
    }catch (std::exception& e)
    {
        m_error = std::format("postprocess error: {}", e.what());
        return false;
    }
}

cv::Rect2i Location::de_letterbox(const cv::Rect2i&& box) noexcept
{
    const double transformedWidth = std::round(m_width * m_scale);
    const double transformedHeight = std::round(m_height * m_scale);
    const double pad_w = (m_letter_width - transformedWidth) / 2.0;
    const double pad_h = (m_letter_height - transformedHeight) / 2.0;
    int x1 = static_cast<int>(std::round((box.x - pad_w) / m_scale));
    int y1 = static_cast<int>(std::round((box.y - pad_h) / m_scale));
    int w = static_cast<int>(std::round(box.width / m_scale));
    int h = static_cast<int>(std::round(box.height / m_scale));
    x1 = std::clamp(x1, 0, m_width - 1);
    y1 = std::clamp(y1, 0, m_height - 1);
    w = std::clamp(w, 0, m_width - x1);
    h = std::clamp(h, 0, m_height - y1);
    cv::Rect rect{ x1, y1, w, h };
    return rect;
}

bool Location::serialize()
{
    const auto last_dot = m_model_path.find_last_of('.');
    std::string prefix;
    if (last_dot != std::string::npos)
    {
        prefix = m_model_path.substr(0, last_dot);
    }else
    {
        throw std::runtime_error("Invalid model path.");
    }
    const auto onnx_path = std::format("{}.onnx", prefix);
    const auto engine_path = std::format("{}.engine", prefix);
    if (std::ifstream engine_i_file(engine_path); engine_i_file && engine_i_file.is_open())
    {
        engine_i_file.close();
        return true;
    }
    std::ifstream onnx_i_file(onnx_path);
    if (!onnx_i_file)
    {
        m_error =  std::format("File not found: {}", onnx_path);
        return false;
    }
    if (onnx_i_file.is_open())
    {
        onnx_i_file.close();
    }
    NvLogger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder)
    {
        m_error = "Failed to create TensorRT builder.";
        return false;
    }
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
    if (!network)
    {
        destroy_builder(builder, nullptr,network);
        m_error = "Failed to create TensorRT network.";
        return false;
    }
    auto* parser = nvonnxparser::createParser(*network,logger);
    if (!parser->parseFromFile(onnx_path.c_str(),static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        destroy_builder(builder, nullptr, network);
        m_error = "Failed to create TensorRT parser.";
        return false;
    }
    auto* config =  builder->createBuilderConfig();
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto* host_memory = builder->buildSerializedNetwork(*network,*config);
    std::ofstream engine_o_file(engine_path,std::ios::binary);
    engine_o_file.write(static_cast<const char*>(host_memory->data()), host_memory->size());
    engine_o_file.close();
    destroy_builder(builder, config, network);
    delete parser;
    return true;
}

size_t Location::get_size_by_dims(const nvinfer1::Dims& dims)
{
    auto size = 1;
    for (auto i{0}; i < dims.nbDims; ++i)
    {
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}

void Location::print_tensor_info(const nvinfer1::ICudaEngine* engine)
{
    const auto tensor_num = engine->getNbIOTensors();
    for (auto i{0}; i < tensor_num; ++i)
    {
        const auto tensor_name = engine->getIOTensorName(i);
        const auto tensor_des = engine->getTensorFormatDesc(tensor_name);
        std::cout<< std::format("{}: {}", tensor_name, tensor_des) << std::endl;
    }
}

void Location::destroy_builder(const nvinfer1::IBuilder* builder, const nvinfer1::IBuilderConfig* config,
    const nvinfer1::INetworkDefinition* network)
{
    if (network) delete network;
    if (config) delete config;
    if (builder) delete builder;
}
