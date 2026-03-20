//
// Created by houmin on 2026/1/16.
//
module;
#include <cmath>
#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <thread>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <QString>

export module copier;

import logger;
import detecter;
import threadpool;
import translate;

using namespace indicators;



/**
 * @brief 配置结构体
 * @param src_dir 源目录
 * @param dst_dir 目标目录
 * @param open_crop 是否开启裁剪
 * @param model 模型路径
 * @param extract_config_dir 提取配置目录
 * @param concurrency_number 并发数量
 */
struct Config
{
    std::vector<std::string> src_dir;
    std::vector<std::string> dst_dir;
    bool open_crop{ false };
    std::string model;
    std::string extract_config_dir;
    int concurrency_number{ 1 };
};

/**
 * @brief 加载配置文件
 * @param path 配置文件路径，默认"./config.json"
 * @return Config 配置结构体
 */
static Config load_config(const std::string_view path = "./config.json")
{
    Config config;
    std::ifstream ifs(path.data());
    if (!ifs.is_open())
    {
        Error("Failed to open config file: {}", path.data());
        return config;
    }
    nlohmann::json j;
    ifs >> j;
    if (j.contains("srcDir") && j["srcDir"].is_array())
    {
        config.src_dir = j["srcDir"].get<std::vector<std::string>>();
    }
    else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: src_dir");
        Error("{}", msg);
        return config;
    }

    if (j.contains("dstDir") && j["dstDir"].is_array())
    {
        config.dst_dir = j["dstDir"].get<std::vector<std::string>>();
    }
    else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: dst_dir");
        Error("{}", msg);
        return config;
    }
    if (j.contains("openCrop") && j["openCrop"].is_boolean())
    {
        config.open_crop = j["openCrop"].get<bool>();
    }
    else
    {
        config.open_crop = false;
    }
    if (j.contains("model") && j["model"].is_string())
    {
        config.model = j["model"].get<std::string>();
    }
    else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: model");
        Error("{}", msg);
        return config;
    }
    if (j.contains("extractConfigDir") && j["extractConfigDir"].is_string())
    {
        config.extract_config_dir = j["extractConfigDir"].get<std::string>();
    }
    else {
        const auto msg = std::string("Missing or incorrect fields in configuration: extractConfigDir");
        Error("{}", msg);
        return config;
    }
    if (j.contains("concurrencyNumber") && j["concurrencyNumber"].is_number_integer())
    {
        config.concurrency_number = j["concurrencyNumber"].get<int>();
    }
    else {
        config.concurrency_number = 1;
    }
    return config;
}

/**
* @brief 提取参数
* @param left_top 左上角点
* @param width 标签宽度
* @param height 标签高度
* @param side_length 二维码边长
*/
struct ExtractParam {
    cv::Point2f left_top;
    float width;
    float height;
    float side_length;
};


/**
* @brief 标签坐标
* @param qr 二维码区域
* @param qr_context 二维码内容
* @param label 整个标签区域
* @param euclidian_distance 欧式距离
*/
struct LabelCoordinates {
    cv::Rect qr;
    std::string qr_context;
    cv::Rect label;
    double euclidian_distance;
};

/**
 * @brief 目录下文件信息
 * @param count 文件数量
 * @param label_name 标签名称
 * @param src_last_dir 源目录最后一级目录路径
 * @param src_last_rename_dir 源目录重命名后的最后一级目录路径
 * @param dst_last_dir 目标目录最后一级目录路径
 * @param file_names 文件名称列表
 */
struct DirFiles
{
    int count{ 0 };
    std::string label_name;
    std::filesystem::path src_last_dir;
    std::filesystem::path src_last_rename_dir;
    std::filesystem::path dst_last_dir;
    std::vector<std::string> file_names;
};

/**
 * @brief 拷贝目录信息
 * @param count 拷贝总的文件数量
 * @param src_root 源目录根路径
 * @param dst_root 目标目录根路径
 * @param dir_files 目录文件信息列表
 */
struct CopyDirInfo
{
    int count{ 0 };
    std::filesystem::path src_root;
    std::filesystem::path dst_root;
    std::vector<DirFiles> dir_files;
};

/**
 * @brief 拷贝器类
 */
export class Copier
{
public:
    explicit Copier();
    ~Copier() = default;
    /**
     * @brief 拷贝文件 同时会进行图片裁剪
     */
    void copy();

private:
    Config m_config{};
    size_t m_threadCount{ 0 };
    std::vector<CopyDirInfo> m_copyFileInfos;
    bool is_ok{ false };
    std::unordered_map<std::string, std::string> m_extract_pinyin;
    std::unordered_map<std::string, ExtractParam> m_extract_params;
    void collect_files();
    auto load_extract_params(std::string_view label_name) -> bool;
    auto load_extract_pinyin() -> bool;
};

struct TaskParam {
    std::filesystem::path srcPath;
    std::filesystem::path srcRenamePath;
    std::filesystem::path dstPath;
    std::vector<std::string> fileNames;
    ExtractParam extractParam;
    bool isFirst{ true };
    Recognize recognize;
    Location<float> location;
    bool isInitialize{ false };
    explicit TaskParam(const std::filesystem::path& src_path,
        const std::filesystem::path& src_rename_path,
        const std::filesystem::path& dst_path,
        std::vector<std::string>&& file_names,
        const ExtractParam& param,
        const bool is_first,
        const std::string& model_path)
        : srcPath(src_path), srcRenamePath(src_rename_path), dstPath(dst_path)
        , fileNames(std::move(file_names)), extractParam(param), isFirst(is_first)
        , recognize(), location(model_path)
    {
        if (location.build()) {
            isInitialize = true;
        }
    }
};

inline static bool transformer_coordinates(const cv::Rect2i& box, const QrCodeResult& qrret,
    const ExtractParam& param, float& dpi, const int width, const int height, LabelCoordinates& lc,
    const int col_offset = 0, const int row_offset = 0) {
    const cv::Point2i p0{ box.x, box.y };
    cv::Point2i qr_left_top_point;
    auto euc_fn = [](int a, int b) {
        return std::sqrt(std::pow(a, 2) + std::pow(b, 2));
        };
    switch (qrret.orientation) {
    case QrOrientation::UP:
    {
        const auto qr_pixel = std::abs(qrret.RightTop.x - qrret.LeftTop.x);
        if (dpi == 0.0) {
            dpi = qr_pixel * 25.4 / param.side_length;
        }
        else {
            dpi = (dpi + qr_pixel * 25.4 / param.side_length) / 2.0;
        }
        const auto temp = dpi / 25.4;
        const auto label_width = param.width * temp;
        const auto label_height = param.height * temp;
        const auto label_x_distance = param.left_top.x * temp;
        const auto label_y_distance = param.left_top.y * temp;
        qr_left_top_point.x = p0.x + qrret.LeftTop.x;
        qr_left_top_point.y = p0.y + qrret.LeftTop.y;
        const auto label_left_top_x = qr_left_top_point.x - label_x_distance;
        const auto label_left_top_y = qr_left_top_point.y - label_y_distance;
        lc.label.x = static_cast<int>(label_left_top_x + 0.5) + col_offset;
        lc.label.y = static_cast<int>(label_left_top_y + 0.5) + row_offset;
        lc.label.width = static_cast<int>(label_width + 0.5);
        lc.label.height = static_cast<int>(label_height + 0.5);
        if (lc.label.x < 0 || lc.label.y < 0) {
            lc.euclidian_distance = euc_fn(width, height);
        }
        else {
            lc.euclidian_distance = euc_fn(lc.label.x, lc.label.y);
        }
        lc.qr.x = p0.x + qrret.LeftTop.x;
        lc.qr.y = p0.y + qrret.LeftTop.y;
        lc.qr.width = qr_pixel;
        lc.qr.height = qr_pixel;
        lc.qr_context = qrret.context;
        return true;
    }
    case QrOrientation::RIGHT:
    {
        const auto qr_pixel = std::abs(qrret.RightTop.y - qrret.LeftTop.y);
        if (dpi == 0.0) {
            dpi = qr_pixel * 25.4 / param.side_length;
        }
        else {
            dpi = (dpi + qr_pixel * 25.4 / param.side_length) / 2.0;
        }
        const auto temp = dpi / 25.4;
        const auto label_width = param.height * temp;
        const auto label_height = param.width * temp;
        const auto label_x_distance = param.left_top.y * temp;
        const auto label_y_distance = param.left_top.x * temp;
        qr_left_top_point.x = p0.x + qrret.LeftTop.x;
        qr_left_top_point.y = p0.y + qrret.LeftTop.y;
        const auto label_left_top_x = qr_left_top_point.x + label_x_distance - label_width;
        const auto label_left_top_y = qr_left_top_point.y - label_y_distance;
        lc.label.x = static_cast<int>(label_left_top_x + 0.5) + col_offset;
        lc.label.y = static_cast<int>(label_left_top_y + 0.5) + row_offset;
        lc.label.width = static_cast<int>(label_width + 0.5);
        lc.label.height = static_cast<int>(label_height + 0.5);
        if (lc.label.x < 0 || lc.label.y < 0) {
            lc.euclidian_distance = euc_fn(width, height);
        }
        else {
            lc.euclidian_distance = euc_fn(lc.label.x, lc.label.y);
        }
        lc.qr.x = p0.x + qrret.LeftBottom.x;
        lc.qr.y = p0.y + qrret.LeftBottom.y;
        lc.qr.width = qr_pixel;
        lc.qr.height = qr_pixel;
        lc.qr_context = qrret.context;
        return true;
    }
    case QrOrientation::DOWN:
    {
        const auto qr_pixel = std::abs(qrret.RightTop.x - qrret.LeftTop.x);
        if (dpi == 0.0) {
            dpi = qr_pixel * 25.4 / param.side_length;
        }
        else {
            dpi = (dpi + qr_pixel * 25.4 / param.side_length) / 2.0;
        }
        const auto temp = dpi / 25.4;
        const auto label_width = param.width * temp;
        const auto label_height = param.height * temp;
        const auto label_x_distance = param.left_top.x * temp;
        const auto label_y_distance = param.left_top.y * temp;
        qr_left_top_point.x = p0.x + qrret.LeftTop.x;
        qr_left_top_point.y = p0.y + qrret.LeftTop.y;
        const auto label_left_top_x = qr_left_top_point.x + label_x_distance - label_width;
        const auto label_left_top_y = qr_left_top_point.y + label_y_distance - label_height;
        lc.label.x = static_cast<int>(label_left_top_x + 0.5) + col_offset;
        lc.label.y = static_cast<int>(label_left_top_y + 0.5) + row_offset;
        lc.label.width = static_cast<int>(label_width + 0.5);
        lc.label.height = static_cast<int>(label_height + 0.5);
        if (lc.label.x < 0 || lc.label.y < 0) {
            lc.euclidian_distance = euc_fn(width, height);
        }
        else {
            lc.euclidian_distance = euc_fn(lc.label.x, lc.label.y);
        }
        lc.qr.x = p0.x + qrret.RightBottom.x;
        lc.qr.y = p0.y + qrret.RightBottom.y;
        lc.qr.width = qr_pixel;
        lc.qr.height = qr_pixel;
        lc.qr_context = qrret.context;
        return true;
    }
    case QrOrientation::LEFT:
    {
        const auto qr_pixel = std::abs(qrret.LeftTop.y - qrret.RightTop.y);
        if (dpi == 0.0) {
            dpi = qr_pixel * 25.4 / param.side_length;
        }
        else {
            dpi = (dpi + qr_pixel * 25.4 / param.side_length) / 2.0;
        }
        const auto temp = dpi / 25.4;
        const auto label_width = param.height * temp;
        const auto label_height = param.width * temp;
        const auto label_x_distance = param.left_top.y * temp;
        const auto label_y_distance = param.left_top.x * temp;
        qr_left_top_point.x = p0.x + qrret.LeftTop.x;
        qr_left_top_point.y = p0.y + qrret.LeftTop.y;
        const auto label_left_top_x = qr_left_top_point.x - label_x_distance;
        const auto label_left_top_y = qr_left_top_point.y + label_y_distance - label_width;
        lc.label.x = static_cast<int>(label_left_top_x + 0.5) + col_offset;
        lc.label.y = static_cast<int>(label_left_top_y + 0.5) + row_offset;
        lc.label.width = static_cast<int>(label_width + 0.5);
        lc.label.height = static_cast<int>(label_height + 0.5);
        if (lc.label.x < 0 || lc.label.y < 0) {
            lc.euclidian_distance = euc_fn(width, height);
        }
        else {
            lc.euclidian_distance = euc_fn(lc.label.x, lc.label.y);
        }
        lc.qr.x = p0.x + qrret.RightTop.x;
        lc.qr.y = p0.y + qrret.RightTop.y;
        lc.qr.width = qr_pixel;
        lc.qr.height = qr_pixel;
        lc.qr_context = qrret.context;
        return true;
    }
    default:
        return false;
    }
}

/// <summary>
/// 滑动区域
/// </summary>
struct SlideArea {
    int sx{ 0 };
    int ex{ 0 };
    int sy{ 0 };
    int ey{ 0 };
};


inline static std::string find_id(const std::string& qr_context) {
    auto pos = qr_context.find("?");
    std::string code_id;
    if (pos != std::string::npos) {
        code_id = qr_context.substr(pos + 1);   // 从 '?' 后一位到结尾
    }
    else {
        pos = qr_context.find("=");
        if (pos != std::string::npos) {
            code_id = qr_context.substr(pos + 1);   // 从 '=' 后一位到结尾
        }
        else {
            if (qr_context.starts_with("http")) {
                return "";
            }
            else {
                return qr_context;
            }
        }
    }
    return code_id;
}

class Task {
    std::unique_ptr<TaskParam> m_param; // 外部参数
    cv::Mat m_matStitch;  // 拼接图像
    cv::Mat m_matEnd;  // 图片尾部
    float m_dpi{ 0.0f };  // DPI
    LabelCoordinates m_labelFirstLineMin;  // 第一行最小标签
    LabelCoordinates m_labelSecondLineAny;  // 第二行任意标签
    LabelCoordinates m_currentLc;  // 动态当前行
    LabelCoordinates m_previousLc;  // 前一行
    bool m_isFoundValidLabel{ false };  // 是否找到有效标签
    int m_meanHeight{ 0 };  // 平均高度
    int m_meanWidth{ 0 }; // 平均宽度
    int m_stepVertical{ 0 };  // 垂直步进
    int m_stepHorizontal{ 0 };  // 水平步进
    int m_lineSapce{ 0 };  // 行距
    int m_splitCount{ 0 };  // 分割数量
    int m_saveCount{ 0 };  // 保存数量



public:
    explicit Task(std::unique_ptr<TaskParam>&& param)
        : m_param{ std::move(param) }
    {
        m_saveCount = m_param->fileNames.size();
    }
    ~Task() = default;

    void run(BlockProgressBar& bar, int subtask_index, const int task_index, const std::string& setting_name,
        std::atomic<int>& actual_count, std::atomic<int>& actual_split_count, std::atomic<int>& bad_count,
        const int total) {
        if (!m_param->isInitialize) {
            Error("Task-{}( {} )_Part-{} initialization failed, skip the task!", task_index, setting_name, subtask_index);
            return;
        }
        Info("Task-{}( {} )_Part-{} start to execute...", task_index, setting_name, subtask_index);
        int file_idx{ 0 };
        bool is_found_valid_params{ false };
        const auto& src_path = m_param->srcPath;
        const auto& file_names = m_param->fileNames;
        const auto& is_first = m_param->isFirst;
        int valid_first_index{ 0 };

        cv::Mat tmp_mat;
        cv::Mat mat_src;

        auto& location = m_param->location;
        auto& recoginze = m_param->recognize;

        const auto& dst_last_dir = m_param->dstPath;

        std::unordered_map<std::string, int> code_id_count; // 二维码ID计数器
        for (auto idx{ 0 }; idx < m_saveCount; ++idx) {
            const auto src_abs_path = src_path / file_names[idx];
            if (tmp_mat.empty()) {
                mat_src = cv::imread(src_abs_path.string());
            }
            else {
                const auto tmp_src = cv::imread(src_abs_path.string());
                if (tmp_src.empty()) {
                    Warn("无法读取图片: {}, 跳过该图片的裁剪操作!", src_abs_path.string());
                    bad_count.fetch_add(1);
                    continue;
                }
                cv::vconcat(tmp_mat, tmp_src, mat_src);
            }
            const auto width = mat_src.cols;
            const auto height = mat_src.rows;
            std::vector<cv::Rect> boxes;
            if (const auto ret = location.infer(mat_src, boxes); !ret) {
                bad_count.fetch_add(1);
                continue;
            }
            if (boxes.empty()) {
                bad_count.fetch_add(1);
                continue;
            }
            const auto boxes_size = boxes.size();
            std::vector<LabelCoordinates> lcs_yes;
            lcs_yes.reserve(boxes_size);
            for (const auto& box : boxes) {
                auto cropped = mat_src(box);
                if (cropped.empty()) {
                    continue;
                }
                //cv::imwrite("cropped.png", cropped);
                if (!cropped.isContinuous()) {
                    cropped = cropped.clone();
                }
                QrCodeResult qr{};
                if (const auto ret = recoginze.detect(cropped, qr); !ret) {
                    continue;
                }
                LabelCoordinates lc{};
                const auto is_ok = transformer_coordinates(box, qr, m_param->extractParam, m_dpi, width, height, lc);
                if (!is_ok) {
                    continue;
                }
                if (lc.label.x < 0 || lc.label.y < 0 || lc.label.x > width || lc.label.y > height || lc.label.x + lc.label.width > width || lc.label.y + lc.label.height > height) {
                    continue;
                }
                //const auto tmp_save_src = mat_src.clone();
                //cv::rectangle(tmp_save_src, lc.label, cv::Scalar(255, 0, 0), 8);
                //cv::imwrite("mat_src.png", tmp_save_src);
                //cv::imshow("cropped", cropped);
                //cv::namedWindow("mat_src", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
                //cv::imshow("mat_src", mat_src);
                //cv::waitKey(0);
                lcs_yes.emplace_back(std::move(lc));
            }
            const auto crop_y = static_cast<int>(static_cast<double>(height) * 3.0 / 4.0 + 0.5);
            tmp_mat = mat_src.rowRange(crop_y, height);
            if (lcs_yes.empty()) {
                bad_count.fetch_add(1);
                continue;
            }
            const auto& file_name = m_param->fileNames[idx];
            const auto dst_abs_path = dst_last_dir / file_name;
            const auto file_stem = dst_abs_path.stem().string();
            const auto file_ext = dst_abs_path.extension().string();
            auto yes_idx{ 0 };
            std::vector<std::string> save_file_names;
            const auto tmp_size = lcs_yes.size();
            for (const auto& item : lcs_yes) {
                const auto save_mat = mat_src(item.label);
                yes_idx++;
                std::string code_id = find_id(item.qr_context);
                if (code_id.empty()) {
                    code_id = std::to_string(yes_idx);
                    const auto save_file_name = std::format("{}_{}{}", file_stem, code_id, file_ext);
                    const auto save_abs_path = (dst_last_dir / save_file_name).string();
                    cv::imwrite(save_abs_path, save_mat);
                    save_file_names.emplace_back(save_file_name);
                    actual_split_count.fetch_add(1);
                }
                else {
                    if (code_id_count.find(code_id) == code_id_count.end()) {
                        code_id_count[code_id] = 1;
                        const auto save_file_name = std::format("{}_{}{}", file_stem, code_id, file_ext);
                        const auto save_abs_path = (dst_last_dir / save_file_name).string();
                        cv::imwrite(save_abs_path, save_mat);
                        save_file_names.emplace_back(save_file_name);
                        actual_split_count.fetch_add(1);
                    }
                    else {
                        code_id_count[code_id]++;
                    }
                }
            }
            std::stringstream ss;
            for (const auto& item : save_file_names) {
                ss << item << ",";
            }
            auto split_path = ss.str();
            if (!split_path.empty() && split_path.back() == ',') {
                split_path.pop_back();
            }
            Info("\nsrc_abs_path: {},\ndst_abs_path: {},\ndst_abs_split_paths: [ {} ]", src_abs_path.string(), dst_abs_path.string(), split_path);
            Logger::flush();

            actual_count.fetch_add(1);
            // 更新进度条
            const auto bar_count = total - bad_count.load(std::memory_order_relaxed);
            bar.set_option(option::MaxProgress{ bar_count });
            bar.set_option(option::PostfixText{ std::format("{}({})/{}, S:{}",
                actual_count.load(std::memory_order_relaxed),
                actual_split_count.load(std::memory_order_relaxed),
                bar_count,
                total) });
            bar.tick();
        }
        Info("Task-{}( {} )_Part-{} finished. Should save count: {}, Actual save count: {}, Actual save split count: {}",
            task_index, setting_name, subtask_index, m_saveCount, file_idx, actual_count.load(std::memory_order_relaxed));
    }
};


Copier::Copier()
    :m_config{ load_config() }
{
    if (m_config.src_dir.empty())
    {
        Error("配置文件没有加载成功！");
        return;
    }
    m_threadCount = m_config.concurrency_number;
    if (m_config.src_dir.size() != m_config.dst_dir.size()) {
        Error("源目录和目标目录配置错误，长度不一致！");
        return;
    }
    const auto config_size = m_config.src_dir.size();
    m_copyFileInfos.reserve(config_size);
    for (auto i{ 0 }; i < config_size; ++i) {
        CopyDirInfo info;
        info.src_root = std::filesystem::path(m_config.src_dir[i]);
        info.dst_root = std::filesystem::path(m_config.dst_dir[i]);
        m_copyFileInfos.emplace_back(std::move(info));
    }
    is_ok = true;
}


std::vector<std::vector<std::string>> split_with_overlap(const std::vector<std::string>& input, const size_t n)
{
    std::vector<std::vector<std::string>> result;
    if (n == 0 || input.empty()) return result;
    result.reserve(n);
    const auto total = input.size();
    const auto base_size = static_cast<int>(std::ceil(static_cast<double>(total) / static_cast<double>(n))) + 1;
    int start{ 0 };
    int end{ 0 };
    for (auto i{ 0 }; i < n; ++i) {
        start = i == 0 ? i * base_size : i * (base_size - 1);
        end = start + base_size;
        if (end > total) {
            end = total;
            i = n; // 结束循环
        }
        result.emplace_back(input.begin() + start, input.begin() + end);
    }
    return result;
}

void Copier::copy()
{
    if (!is_ok)
    {
        const auto msg = "拷贝器没有初始化成功，无法进行拷贝操作!";
        std::cout << msg << std::endl;
        Info("{}", msg);
        return;
    }
    if (const auto ret = load_extract_pinyin(); !ret) {
        const auto msg = "加载提取参数失败，无法进行拷贝操作!";
        std::cout << msg << std::endl;
        Info("{}", msg);
        return;
    }
    collect_files();
    auto task_count = 0;
    auto copy_count = 0;
    {
        std::stringstream ss_src;
        std::stringstream ss_dst;
        for (const auto& info : m_copyFileInfos) {
            task_count += info.dir_files.size();
            copy_count += info.count;
            ss_src << info.src_root.string() << "; ";
            ss_dst << info.dst_root.string() << "; ";
        }
        const auto msg = std::format("拷贝的任务总数: {}\n拷贝文件的数量: {}\n拷贝目录的源文件根目录: {}\n拷贝目录的目标文件根目录: {}\n",
            task_count, copy_count, ss_src.str(), ss_dst.str());
        std::cout << msg << std::endl;
        Info("{}", msg);
    }
    if (task_count == 0 || copy_count == 0)
    {
        const auto msg = "没有要拷贝的图片数据!";
        std::cout << msg << std::endl;
        Info("{}", msg);
        return;
    }
    const std::array<Color, 7> colors{ Color::grey, Color::red, Color::green, Color::yellow, Color::blue,Color::magenta, Color::cyan };

    ThreadPool pool{ m_threadCount };
    std::cout << std::endl << std::endl;
    int task_index{ 1 };
    indicators::show_console_cursor(false);

    for (const auto& copy_file_info : m_copyFileInfos) {
        for (const auto& [count, label_name, src_last_dir,
            src_last_rename_dir, dst_last_dir, file_names] : copy_file_info.dir_files)
        {
            if (!load_extract_params(label_name)) {
                const auto msg = std::format("加载标签[ {} ]的配置失败!", label_name);
                std::cerr << msg << std::endl;
                Warn("{}", msg);
                continue;
            }
            if (!std::filesystem::exists(dst_last_dir))
            {
                std::filesystem::create_directories(dst_last_dir);
            }
            {
                const auto msg = std::format("<============== [ Task-{}( {} ) 开始拷贝... ] ==============>\n标签名称: {}, 文件数量: {}\n源目录: {}\n目标目录: {}",
                    task_index, label_name, label_name, count, src_last_dir.string(), dst_last_dir.string());
                std::cout << msg << std::endl;
                Info("{}", msg);
            }
            int index{ 0 };
            const auto now = std::chrono::system_clock::now();
            const auto now_str = std::format("{:%Y-%m-%d %H:%M:%S}", now);
            {
                const auto msg = std::format("拷贝开始时间: {}", now_str);
                std::cout << msg << std::endl;
                Info("{}", msg);
            }
            BlockProgressBar bar{
                option::BarWidth{80},
                option::ForegroundColor{Color::green},
                option::ShowPercentage{true},
                option::ShowElapsedTime{true},
                option::FontStyles{std::vector{FontStyle::bold}},
                option::MaxProgress{count}
            };
            bar.set_option(option::PostfixText{ std::format("{}/{}",0, count) });
            const auto split_file_names = split_with_overlap(file_names, m_threadCount);
            const auto split_count = split_file_names.size();
            const auto repeat_count = split_count - 1;
            std::vector<std::future<void>> results;
            results.reserve(split_count);
            std::atomic<int> bad_count{ 0 };
            std::atomic<int> actual_count{ 0 };
            std::atomic<int> actual_split_count{ 0 };
            for (auto i{ 0 }; i < split_count; ++i) {
                std::vector<std::string> temp_file_names = split_file_names[i];
                const auto temp_subtask_index = i;
                const auto temp_src_last_dir = src_last_dir;
                const auto temp_src_rename_dir = src_last_rename_dir;
                const auto temp_dst_last_dir = dst_last_dir;
                const auto temp_extract_params = m_extract_params[label_name];
                const auto temp_label_name = label_name;
                const auto temp_model = m_config.model;
                results.emplace_back(pool.submit(
                    [temp_file_names = std::move(temp_file_names),
                    &bar, temp_subtask_index,
                    temp_src_last_dir, temp_src_rename_dir, temp_dst_last_dir,
                    temp_extract_params, temp_label_name, temp_model, task_index,
                    &actual_count, &actual_split_count, &bad_count, temp_total = count]() mutable {
                        const auto& t1 = temp_file_names;
                        const auto flag = temp_subtask_index == 0 ? true : false;
                        auto tkp = std::make_unique<TaskParam>(
                            temp_src_last_dir, temp_src_rename_dir, temp_dst_last_dir, std::move(temp_file_names),
                            temp_extract_params, flag, temp_model);
                        Task tk{ std::move(tkp) };
                        tk.run(bar, temp_subtask_index, task_index, temp_label_name,
                            actual_count, actual_split_count, bad_count, temp_total);
                    }
                ));
            }
            for (auto& result : results) {
                result.get();
            }
            bar.mark_as_completed();
            const auto tmp_actual_count = actual_count.load(std::memory_order_relaxed) - repeat_count;
            const auto tmp_bad_count = bad_count.load(std::memory_order_relaxed);
            const auto tmp_actual_split_count = actual_split_count.load(std::memory_order_relaxed);
            {
                const auto msg = std::format("汇总: 应拷贝数量: {}, 实际拷贝数量: {}, 错误图片数量: {}, 实际拆分拷贝数量: {}",
                    count, tmp_actual_count, tmp_bad_count,
                    tmp_actual_split_count);
                std::cout << msg << std::endl;
                Info("{}", msg);
            }
            if (count == (tmp_bad_count + tmp_actual_count)) {
                std::filesystem::rename(src_last_dir, src_last_rename_dir);
                {
                    const auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::system_clock::now());
                    const auto msg = std::format("拷贝结束时间: {}\n文件拷贝完成, 并重命名源目录: {}\n<============== [ Task-{}( {} ) 拷贝完成! ] ==============>",
                        time_str, src_last_rename_dir.string(), task_index++, label_name);
                    std::cout << msg << std::endl;
                    Info("{}", msg);
                }
                std::cout << std::endl << std::endl;
            }
            else {
                const auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::system_clock::now());
                const auto msg = std::format("拷贝结束时间: {}\n文件拷贝失败 \n<============== [ Task-{}( {} ) 拷贝失败，请核对! ] ==============>",
                    time_str, task_index++, label_name);
                std::cout << msg << std::endl;
                Info("{}", msg);
                std::cout << std::endl << std::endl;
            }
        }
    }

    indicators::show_console_cursor(true);
    {
        const auto msg = "所有文件拷贝完成!";
        std::cout << msg << std::endl;
        Info("{}", msg);
    }
}

static long long extract_number(const std::string& s, bool& ok)
{
    static const std::regex re(R"((\d+))");
    if (std::smatch m; std::regex_search(s, m, re) && m.size() >= 2)
    {
        ok = true;
        return std::stoll(m.str(1));
    }
    ok = false;
    return 0;
}

static void sort_files(std::vector<std::string>& files)
{
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b)
        {
            bool ok_a{ false };
            bool ok_b{ false };
            const auto num_a = extract_number(a, ok_a);
            const auto num_b = extract_number(b, ok_b);
            if (ok_a && ok_b) return num_a < num_b;
            if (ok_a != ok_b) return ok_a;
            return a < b;
        });
}

void Copier::collect_files()
{
    for (auto& copy_file_info : m_copyFileInfos) {
        int sum{ 0 };
        for (const auto& date_entry : std::filesystem::directory_iterator(copy_file_info.src_root))
        {
            if (!date_entry.is_directory()) continue;
            const std::string date_dir_name = date_entry.path().filename().string();
            for (const auto& label_entry : std::filesystem::directory_iterator(date_entry))
            {
                if (!label_entry.is_directory()) continue;
                if (label_entry.path().filename().string().ends_with("_copied")) continue;
                const std::string label_dir_name = label_entry.path().filename().string();
                std::string label_name;
                const auto find = label_dir_name.rfind('_');
                if (find != std::string::npos) {
                    label_name = label_dir_name.substr(0, find);
                }
                else {
                    label_name = label_dir_name;
                }
                const std::string label_dir_rename = std::format("{}_copied", label_dir_name);
                int count{ 0 };
                std::vector<std::string> file_names;
                for (const auto& file_entry : std::filesystem::directory_iterator(label_entry))
                {
                    if (!file_entry.is_regular_file()) continue;
                    const auto ext = file_entry.path().extension().string();
                    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp" && ext != ".tiff") continue;
                    file_names.emplace_back(file_entry.path().filename().string());
                    ++count;
                }
                sum += count;
                const std::filesystem::path src_last_dir = copy_file_info.src_root / date_dir_name / label_dir_name;
                const std::filesystem::path src_last_rename_dir = copy_file_info.src_root / date_dir_name / label_dir_rename;
                const std::filesystem::path dst_last_dir = copy_file_info.dst_root / date_dir_name / label_dir_name;
                sort_files(file_names);
                copy_file_info.dir_files.emplace_back(DirFiles{ count,label_name,src_last_dir,src_last_rename_dir,
                    dst_last_dir,std::move(file_names) });
            }
        }
        copy_file_info.count = sum;
    }
}


auto Copier::load_extract_pinyin() -> bool {
    bool flag{ false };
    for (const auto& config_entry : std::filesystem::directory_iterator(m_config.extract_config_dir)) {
        if (!config_entry.is_regular_file()) continue;
        const std::string file_name = config_entry.path().stem().string();
        const std::string pinyin_file_name = qStringToPinYin(QString::fromStdString(file_name));
        m_extract_pinyin[pinyin_file_name] = file_name;
        flag = true;
    }
    return flag;
}


auto Copier::load_extract_params(const std::string_view label_name)->bool
{
    if (auto it = m_extract_params.find(label_name.data()); it != m_extract_params.end()) {
        return true;
    }
    std::filesystem::path extract_config_path(m_config.extract_config_dir);
    try {
        extract_config_path /= std::format("{}.json", m_extract_pinyin[label_name.data()]);
    }
    catch (std::exception& e) {
        Warn("文件夹拼音名称: {} 对应的标签配置未找到: {}", label_name, e.what());
        return false;
    }

    if (!std::filesystem::exists(extract_config_path)) {
        Warn("文件路径: {} 未找到!", extract_config_path.string());
        return false;
    }
    std::ifstream ifs(extract_config_path.string());
    if (!ifs.is_open())
    {
        const auto msg = std::format("Failed to open config file: {}", extract_config_path.string());
        Error("{}", msg);
        return false;
    }
    nlohmann::json j;
    ifs >> j;
    ExtractParam param{};
    if (j.contains("extract_x"))
    {
        param.left_top.x = j["extract_x"].get<float>();
    }
    else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: extract_x");
        Error("{}", msg);
        return false;
    }
    if (j.contains("extract_y")) {
        param.left_top.y = j["extract_y"].get<float>();
    }
    else {
        const auto msg = std::string("Missing or incorrect fields in configuration: extract_y");
        Error("{}", msg);
        return false;
    }
    if (j.contains("label_width")) {
        param.width = j["label_width"].get<float>();
    }
    else {
        const auto msg = std::string("Missing or incorrect fields in configuration: label_width");
        Error("{}", msg);
        return false;
    }
    if (j.contains("label_height")) {
        param.height = j["label_height"].get<float>();
    }
    else {
        const auto msg = std::string("Missing or incorrect fields in configuration: label_width");
        Error("{}", msg);
        return false;
    }
    if (j.contains("extract_sideLength")) {
        param.side_length = j["extract_sideLength"].get<float>();
    }
    else {
        const auto msg = std::string("Missing or incorrect fields in configuration: extract_sideLength");
        Error("{}", msg);
        return false;
    }
    m_extract_params[std::string(label_name.data())] = param;
    return true;
}
