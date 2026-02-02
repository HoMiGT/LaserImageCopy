//
// Created by houmin on 2026/1/16.
//
module;
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

export module copier;

import logger;
import detecter;
import threadpool;

using namespace indicators;



/**
 * @brief 配置结构体
 * @param src_dir 源目录
 * @param dst_dir 目标目录
 * @param open_crop 是否开启裁剪
 * @param model 模型路径
 * @param extract_config_dir 提取配置目录
 */
struct Config
{
    std::string src_dir;
    std::string dst_dir;
    bool open_crop{false};
    std::string model;
    std::string extract_config_dir;
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
    if (j.contains("srcDir") && j["srcDir"].is_string())
    {
        config.src_dir = j["srcDir"].get<std::string>();
    }else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: src_dir");
        Error("{}", msg);
        return config;
    }

    if (j.contains("dstDir")&&j["dstDir"].is_string())
    {
        config.dst_dir = j["dstDir"].get<std::string>();
    }else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: dst_dir");
        Error("{}",msg);
        return config;
    }
    if (j.contains("openCrop") && j["openCrop"].is_boolean())
    {
        config.open_crop = j["openCrop"].get<bool>();
    }else
    {
        config.open_crop=false;
    }
    if (j.contains("model") && j["model"].is_string())
    {
        config.model = j["model"].get<std::string>();
    }else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: model");
        Error("{}",msg);
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
    int count{0};
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
    int count{0};
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
    ~Copier()=default;
    /**
     * @brief 拷贝文件 同时会进行图片裁剪
     */
    void copy();

private:
    Config m_config{};
    size_t m_threadCount{0};
    CopyDirInfo m_copyFileInfo;
    bool is_ok{ false };
    std::unordered_map<std::string, ExtractParam> m_extract_params;
    void collect_files();
    auto load_extract_params(std::string_view label_name)->bool;
};

struct TaskParam {
    std::filesystem::path srcPath;
    std::filesystem::path srcRenamePath;
    std::filesystem::path dstPath;
    std::vector<std::string> fileNames;
	ExtractParam extractParam;
	bool isFirst{ true };
	Recognize recognize;
    Location location;
    bool isInitialize{ false };
    explicit TaskParam(const std::filesystem::path& src_path,
        const std::filesystem::path& src_rename_path,
        const std::filesystem::path& dst_path,
        std::vector<std::string>&& file_names,
        const ExtractParam& param,
        const bool is_first,
        const std::string& model_path) 
        : srcPath(src_path),srcRenamePath(src_rename_path),dstPath(dst_path)
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
        const auto qr_pixel = qrret.RightTop.x - qrret.LeftTop.x;
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
        const auto qr_pixel = qrret.RightTop.y - qrret.LeftTop.y;
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
        const auto label_left_top_x = qr_left_top_point.x + label_x_distance - label_height;
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
        const auto qr_pixel = qrret.RightTop.x - qrret.LeftTop.x;
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
        const auto qr_pixel = qrret.LeftTop.y - qrret.RightTop.y;
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

    void find_qualified_labels(){
        m_isFoundValidLabel = false;
		const auto image_width = m_matStitch.cols;
        const auto image_height = m_matStitch.rows;
        const auto max_length = std::max(image_width, image_height);
        auto m_row_rate = 0.25 * max_length / image_height;

        auto& location = m_param->location;
        auto& recoginze = m_param->recognize;
        int max_row_count{ 0 };
        int max_col_count{ 0 };
        const std::vector<double> col_rates{0.33};
        for (const auto& col_rate : col_rates) {
            while (true) {
                auto crop_width = static_cast<int>(image_width * col_rate + m_stepHorizontal);
                crop_width = std::clamp(crop_width, 0, image_width);
                auto crop_height = static_cast<int>(image_height * m_row_rate + m_stepVertical);
                crop_height = std::clamp(crop_height, 0, image_height);
                if (crop_width >= image_width) {
                    max_col_count++;
                }
                if (crop_height >= image_height) {
                    max_row_count++;
                }
                if (max_col_count > 1 || max_row_count > 1) {
                    break;
                }
                cv::Rect rect{ 0,0,crop_width, crop_height };
                cv::Mat window = m_matStitch(rect);
                std::vector<cv::Rect> boxes;
                if (const auto ret = location.infer(window, boxes); !ret) {
                    m_row_rate *= 1.41421356;
                    continue;
                }
                if (boxes.empty()) {
                    m_row_rate *= 1.41421356;
                    continue;
                }
                const auto boxes_size = boxes.size();
                std::vector<LabelCoordinates> lcs;
                lcs.reserve(boxes_size);
                for (const auto& box : boxes) {
                    auto cropped = window(box);
                    if (!cropped.isContinuous()) {
                        cropped = cropped.clone();
                    }
                    QrCodeResult qr{};
                    if (const auto ret = recoginze.detect(cropped, qr); !ret) {
                        Warn("{}", recoginze.what());
                        continue;
                    }
                    LabelCoordinates lc{};
                    const auto is_ok = transformer_coordinates(box, qr, m_param->extractParam, m_dpi, crop_width, crop_height, lc);
                    if (!is_ok) {
                        continue;
                    }
                    if (lc.label.x < 0 || lc.label.y < 0 || lc.label.x > crop_width || lc.label.y > crop_height) {
                        continue;
                    }
                    lcs.emplace_back(std::move(lc));
                }
                if (lcs.empty()) {
                    m_row_rate *= 1.41421356;
                    continue;
                }
                auto min_it = std::min_element(
                    lcs.begin(), lcs.end(),
                    [](const LabelCoordinates& a, const LabelCoordinates& b) {
                        return a.euclidian_distance < b.euclidian_distance;
                    }
                );
                if (min_it != lcs.end()) {
                    for (const auto& item : lcs) {
                        if (m_meanHeight == 0) {
                            m_meanHeight = static_cast<int>((min_it->label.height + item.label.height) / 2.0 + 0.5);
                        }
                        else {
                            m_meanHeight = static_cast<int>((m_meanHeight + static_cast<int>((min_it->label.height + item.label.height) / 2.0 + 0.5)) / 2.0 + 0.5);
                        }
                        if (m_meanWidth == 0) {
                            m_meanWidth = static_cast<int>((min_it->label.width + item.label.width) / 2.0 + 0.5);
                        }
                        else {
                            m_meanWidth = static_cast<int>((m_meanWidth + static_cast<int>((min_it->label.width + item.label.width) / 2.0 + 0.5)) / 2.0 + 0.5);
                        }
                        const auto distance = std::abs(min_it->label.y - item.label.y);
                        if (distance > m_meanHeight * 0.8 && distance < m_meanHeight * 2) {
                            m_labelFirstLineMin = *min_it;
                            m_labelSecondLineAny = item;
                            m_currentLc = m_labelFirstLineMin;
                            m_previousLc = m_labelFirstLineMin;
                            m_isFoundValidLabel = true;
                            break;
                        }
                    }
                }
                if (m_isFoundValidLabel) {
                    break;
                }
                if (crop_width >= image_width && crop_height >= image_height) {
                    break;
                }
                m_row_rate *= 1.41421356;
            }
            if (m_isFoundValidLabel) {
                break;
            }
        }
    }

    void slide_capture(SlideArea& sa,const int idx, std::atomic<int>& actual_split_count) {
        // 识别、定位、额外辅助参数
        auto& recognize = m_param->recognize;
        auto& location = m_param->location;
        auto& extract_param = m_param->extractParam;
        // 文件路径以及文件信息
        const auto& dst_last_dir = m_param->dstPath;
		const auto& file_name = m_param->fileNames[idx];
        const auto dst_abs_path = dst_last_dir / file_name;
		const auto file_stem = dst_abs_path.stem().string();
		const auto file_ext = dst_abs_path.extension().string();
        // 保存路径的列表信息
        const auto image_width = m_matStitch.cols;
        const auto image_height = m_matStitch.rows;
        const auto split_count = static_cast<int>(static_cast<double>(image_height) / m_meanHeight + 0.5);
        std::vector<std::string> save_file_names;
        save_file_names.reserve(split_count);
        bool is_invalid_image{ false };
        int slide_idx{ 0 };

        do {
            if (m_param->isFirst && idx == 0 && slide_idx == 0) {
                // 仅文件夹第一个文件保存
                const auto offset = static_cast<int>(m_lineSapce / 2.0 + 0.5);
                const auto cut_start = m_currentLc.label.y - offset;
                const auto cut_end = m_currentLc.label.y + m_currentLc.label.height + offset;
                const auto save_mat = m_matStitch.rowRange(cut_start, cut_end);
                const auto save_abs_path = (dst_last_dir / std::format("{}_P0{}", file_stem, file_ext)).string();
                cv::imwrite(save_abs_path, save_mat);
                save_file_names.emplace_back(save_abs_path);
                actual_split_count.fetch_add(1);
                // 更新参数
                sa.sy += static_cast<int>(m_lineSapce / 2.0 + 0.5) * 2 + m_meanHeight;
                sa.ey += static_cast<int>(m_lineSapce / 2.0 + 0.5) * 2 + m_meanHeight;
                continue;
            }
            if (!m_param->isFirst && idx == 0 && slide_idx == 0) {
                // 其他任务的第一个文件是重复，所以不保存仅更新参数
                sa.sy += static_cast<int>(m_lineSapce / 2.0 + 0.5) * 2 + m_meanHeight;
                sa.ey += static_cast<int>(m_lineSapce / 2.0 + 0.5) * 2 + m_meanHeight;
                continue;
            }
            bool is_found_next{ false };
            int max_row_count{ 0 };
            int max_col_count{ 0 };
            // 滑动区域，检测下一个完整标签
            while (true) {
                auto sx = sa.sx;
                auto sy = sa.sy;
                auto sw = sa.ex - sa.sx;
                auto sh = sa.ey - sa.sy;
                // 限制截取超出边界
                sx = std::clamp(sx, 0, image_width);
                sy = std::clamp(sy, 0, image_height);
                if (sx + sw > image_width) {
                    sw = image_width - sx;
                    max_col_count++;
                }
                if (sy + sh > image_height) {
                    sh = image_height - sy;
                    max_row_count++;
                }
                // 多次区域已经最大，但无结果，说明该图是无用图
                if (max_col_count > 1 && max_row_count > 1) {
                    const auto path = m_param->srcPath / file_name;
                    Warn("请核查, 该图无法正确检测合格标签: {}", path.string());
                    is_invalid_image = true;
                    break;
                }
                // 截取图片
                cv::Rect slide_win{ sx,sy,sw,sh };
                cv::Mat slide_mat = m_matStitch(slide_win);
                // 进行推理检测并调整位置
                std::vector<cv::Rect> boxes;
                if (const auto ret = location.infer(slide_mat, boxes); !ret) {
                    sa.ex += m_meanWidth * 0.618;
                    sa.ey += m_meanWidth * 0.618;
                    continue;
                }
                if (boxes.empty()) {
                    sa.ex += m_meanWidth * 0.618;
                    sa.ey += m_meanWidth * 0.618;
                    continue;
                }
                std::vector<LabelCoordinates> lcs;
                lcs.reserve(boxes.size());
                QrCodeResult qr{};
                for (const auto& box : boxes) {
                    auto cropped = slide_mat(box);
                    if (!cropped.isContinuous())
                        cropped = cropped.clone();
                    if (const auto ret = recognize.detect(cropped, qr); !ret) {
                        Warn("二维码识别异常: {}", recognize.what());
                        continue;
                    }
                    LabelCoordinates lc{};
                    if (const auto ok = transformer_coordinates(box, qr, extract_param, m_dpi, sw, sh, lc, sx, sy); !ok) {
                        continue;
                    }
                    // 不是合格的完整区域的标签
                    if (lc.label.x < 0 || lc.label.y < 0 || lc.label.x > sw + sx
                        || lc.label.y > sh + sy) {
                        continue;
                    }
                    lcs.emplace_back(std::move(lc));
                }
                if (lcs.empty()) {
                    sa.ex += m_meanWidth * 0.618;
                    sa.ey += m_meanWidth * 0.618;
                    continue;
                }
                auto min_it = std::min_element(lcs.begin(), lcs.end(),
                    [](const LabelCoordinates& a, const LabelCoordinates& b) {
                        return a.euclidian_distance < b.euclidian_distance;
                    });
                if (min_it != lcs.end()) {
                    m_currentLc = *min_it;
                    is_found_next = true;
                    break;
                }
                sa.ex += m_meanWidth * 0.618;
                sa.ey += m_meanWidth * 0.618;
            }
            if (is_found_next) {
                Info("二维码信息: {}, 二维码坐标信息: (x: {}, y: {}, w: {}, h: {}), 完整PDF标签坐标信息: (x: {}, y: {}, w: {}, h: {})", 
                    m_currentLc.qr_context,m_currentLc.qr.x, m_currentLc.qr.y, m_currentLc.qr.width, m_currentLc.qr.height,
                    m_currentLc.label.x, m_currentLc.label.y, m_currentLc.label.width, m_currentLc.label.height);
                // 保存图片，并更新参数，并对下一个区域进行判断是否满足下一次的截取
                // 满足则进行下一次循环，
                // 不满足则保存末尾的图片，结束循环，进行下一张图片的拼接和重新截取
                m_meanHeight = m_meanHeight == 0 ? m_currentLc.label.height : static_cast<int>((m_meanHeight + m_currentLc.label.height) / 2.0 + 0.5);
                m_meanWidth = m_meanWidth == 0 ? m_currentLc.label.width : static_cast<int>((m_meanWidth + m_currentLc.label.width) / 2.0 + 0.5);
                if (const auto temp_lineSapce = std::abs(m_previousLc.label.y - m_currentLc.label.y) - m_meanHeight; temp_lineSapce > 0 && temp_lineSapce < m_meanHeight) {
                    m_lineSapce = temp_lineSapce;
                }
                const auto offset = static_cast<int>(m_lineSapce / 2.0 + 0.5);
                auto cut_start = m_currentLc.label.y - offset;
                cut_start = std::clamp(cut_start, 0, image_height);
                auto cut_end = m_currentLc.label.y + m_currentLc.label.height + offset;
                cut_end = std::clamp(cut_end, 0, image_height);
                if (idx > 0 || (idx==0 && m_param->isFirst)) {
                    const auto save_mat = m_matStitch.rowRange(cut_start, cut_end);
                    const auto save_file_name = std::format("{}_P{}{}", file_stem, slide_idx, file_ext);
                    const auto save_abs_path = (dst_last_dir / save_file_name).string();
                    cv::imwrite(save_abs_path, save_mat);
                    save_file_names.emplace_back(save_file_name);
                    actual_split_count.fetch_add(1);
                }
                sa.sx = static_cast<int>(m_currentLc.label.x - m_meanWidth * 0.25 + 0.5);
                sa.ex = static_cast<int>(m_currentLc.label.x + m_meanWidth * 1.25 + 0.5);
                sa.sy += static_cast<int>(m_lineSapce / 2.0 + 0.5) * 2 + m_meanHeight;
                sa.ey += static_cast<int>(m_lineSapce / 2.0 + 0.5) * 2 + m_meanHeight;
                m_previousLc = m_currentLc;
                // 更新图片末尾
                if (sa.sy > image_height) {
                    break;
                }
                if (sa.ey > image_height) {
                    m_matEnd = m_matStitch.rowRange(cut_end, image_height);
                    break;
                }
            }
        } while (!is_invalid_image && slide_idx++ < split_count);
        const auto src_abs_path = m_param->srcPath / file_name;
        std::stringstream ss;
        for (const auto& item : save_file_names) {
            ss << item << ",";
        }
        auto split_path = ss.str();
        if (!split_path.empty() && split_path.back() == ',') {
            split_path.pop_back();
        }
        Info("\nsrc_abs_path: {},\ndst_abs_path: {},\ndst_abs_split_paths: [ {} ]",src_abs_path.string(),dst_abs_path.string(),split_path);
        Logger::flush();
    }


public:
    explicit Task(std::unique_ptr<TaskParam>&& param)
        : m_param{std::move(param)}
    {
        m_saveCount = m_param->fileNames.size();
    }
    ~Task() = default;

    void run(BlockProgressBar& bar, int subtask_index, const int task_index, const std::string& setting_name,
        std::atomic<int>& actual_count, std::atomic<int>& actual_split_count, const int total){
        Info("Task-{}( {} )_Part-{} start to execute...", task_index, setting_name, subtask_index);
        int file_idx{ 0 };
		bool is_found_valid_params{ false };
        const auto& src_path = m_param->srcPath;
		const auto& file_names = m_param->fileNames;
        const auto& is_first = m_param->isFirst;
        int valid_first_index{ 0 };
        do {
			// 初始化滑动区域参数
            SlideArea sa{};
			const auto src_abs_path = src_path / file_names[file_idx];
            const auto mat_src = cv::imread(src_abs_path.string());
            // 读取图片并拼接
            if (file_idx <= valid_first_index ) {
                m_matStitch = mat_src;
            }
            else {
                cv::vconcat(m_matEnd,mat_src,m_matStitch);
            }
            // 查找合适的标签以及标签参数
            find_qualified_labels();
            if (!m_isFoundValidLabel) {
                valid_first_index = file_idx;
                Warn("在图片[ {} ]中没有找到合适的标签，跳过该图片的裁剪操作!", src_abs_path.string());
                continue;
            }
            m_lineSapce = static_cast<int>(std::abs(m_labelSecondLineAny.label.y - m_labelFirstLineMin.label.y) - m_meanHeight + 0.5);
            sa.sx = std::max(0, m_labelFirstLineMin.label.x - static_cast<int>(m_labelFirstLineMin.label.width * 0.25 + 0.5));
            sa.ex = std::min(m_matStitch.cols, m_labelFirstLineMin.label.x + static_cast<int>(m_labelFirstLineMin.label.width * 1.25 + 0.5));
            sa.sy = std::max(0, m_labelFirstLineMin.label.y - static_cast<int>(m_lineSapce * 0.5 + 0.5));
            sa.ey = std::min(m_matStitch.rows, m_labelFirstLineMin.label.y + m_meanHeight + static_cast<int>(m_lineSapce * 0.5 + 0.5));
            // 进行滑动裁剪
            slide_capture(sa, file_idx, actual_split_count);

            if (!is_first && file_idx == 0) {
                continue;
            }
            actual_count.fetch_add(1);
            bar.set_option(option::PostfixText{ std::format("{}({})/{}",actual_count.load(std::memory_order_relaxed),actual_split_count.load(std::memory_order_relaxed), total) });
            bar.tick();
        } while (++file_idx < m_saveCount);
        Info("Task-{}( {} )_Part-{} finished. Should save count: {}, Actual save count: {}, Actual save split count: {}", 
            task_index, setting_name, subtask_index, m_saveCount, file_idx, actual_count.load(std::memory_order_relaxed));
    }
};


Copier::Copier()
    :m_config{load_config()}
{
    if (m_config.src_dir.empty())
    {
		Error("配置文件没有加载成功！");
        return;
    }
    m_threadCount = std::thread::hardware_concurrency();
    m_threadCount = 2;
    m_copyFileInfo.src_root = std::filesystem::path(m_config.src_dir);
    m_copyFileInfo.dst_root = std::filesystem::path(m_config.dst_dir);
    is_ok = true;
}


std::vector<std::vector<std::string>> split_with_overlap(const std::vector<std::string>& input, const size_t n)
{
    std::vector<std::vector<std::string>> result;
    if (n == 0 || input.empty()) return result;

    const auto total = input.size();
    const auto base_size = total / n;
    const auto remainder = total % n;
    auto start = 0;
    for (auto i{ 0 }; i < n; ++i) {
        auto part_size = base_size + (i < remainder ? 1 : 0);
        if (i > 0 && start > 0) {
            --start;
            ++part_size;
        }
        if (start + part_size > total) part_size = total - start;
        result.emplace_back(input.begin() + start, input.begin() + start + part_size);
        start += part_size;
    }
    return result;
}

void Copier::copy()
{
    if (!is_ok)
    {
        const auto msg = "拷贝器没有初始化成功，无法进行拷贝操作!";
        std::cout<< msg << std::endl;
        Info("{}", msg);
        return;
	}
    collect_files();
    {
        const auto msg = std::format("拷贝的任务总数: {}\n拷贝文件的数量: {}\n拷贝目录的源文件根目录: {}\n拷贝目录的目标文件根目录: {}\n",
            m_copyFileInfo.dir_files.size(), m_copyFileInfo.count, m_copyFileInfo.src_root.string(), m_copyFileInfo.dst_root.string());
        std::cout << msg << std::endl;
        Info("{}", msg);

    }
    if (m_copyFileInfo.dir_files.empty())
    {
        const auto msg = "没有要拷贝的图片数据!";
        std::cout << msg << std::endl;
        Info("{}", msg);
        return;
    }
    const std::array<Color,7> colors{ Color::grey, Color::red, Color::green, Color::yellow, Color::blue,Color::magenta, Color::cyan};
    
    ThreadPool pool{ m_threadCount };
    std::cout<< std::endl << std::endl;
    int task_index{1};
    show_console_cursor(false);
    for (const auto& [count, label_name, src_last_dir,
        src_last_rename_dir, dst_last_dir, file_names]: m_copyFileInfo.dir_files)
    {
        if (!load_extract_params(label_name)) {
            const auto msg = std::format("加载标签[ {} ]的配置失败!", label_name);
            std::cerr << msg << std::endl;
            Warn("{}", msg);
            continue;
        }
        {
            const auto msg = std::format("<============== [ Task-{}( {} ) 开始拷贝... ] ==============>\n标签名称: {}, 文件数量: {}\n源目录: {}\n目标目录: {}",
                task_index, label_name, label_name, count, src_last_dir.string(), dst_last_dir.string());
            std::cout << msg << std::endl;
            Info("{}", msg);
        }
        int index{0};
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
        const auto split_file_names =  split_with_overlap(file_names, m_threadCount);
        std::vector<std::future<void>> results;
        results.reserve(m_threadCount);
        std::atomic<int> actual_count{ 0 };
        std::atomic<int> actual_split_count{ 0 };
        for (auto i{ 0 }; i < m_threadCount; ++i) {
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
                 &bar,  temp_subtask_index,
                temp_src_last_dir, temp_src_rename_dir, temp_dst_last_dir,
                temp_extract_params, temp_label_name, temp_model, task_index,
                &actual_count,&actual_split_count,temp_total = count]() mutable {
                    const auto& t1 = temp_file_names;
                    const auto flag = temp_subtask_index == 0 ? true : false;
                    auto tkp = std::make_unique<TaskParam>(
                        temp_src_last_dir, temp_src_rename_dir, temp_dst_last_dir, std::move(temp_file_names),
                        temp_extract_params, flag, temp_model);
                    Task tk{ std::move(tkp) };
                    tk.run(bar, temp_subtask_index, task_index, temp_label_name,actual_count,actual_split_count, temp_total);
                }
            ));
        }
        for (auto& result : results) {
            result.get();
        }
        bar.mark_as_completed();
        {
            const auto msg = std::format("汇总: 应拷贝数量: {}, 实际拷贝数量: {}, 实际拆分拷贝数量: {}", 
                count, actual_count.load(std::memory_order_relaxed), actual_split_count.load(std::memory_order_relaxed));
            std::cout << msg << std::endl;
            Info("{}", msg);
        }
        // 测试先注释掉重命名操作
        // std::filesystem::rename(src_last_dir,src_last_rename_dir);
        {
            const auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::system_clock::now());
            const auto msg = std::format("拷贝结束时间: {}\n文件拷贝完成, 并重命名源目录: {}\n<============== [ Task-{}( {} ) 拷贝完成! ] ==============>", 
                time_str, src_last_rename_dir.string(), task_index++, label_name);
            std::cout << msg << std::endl;
            Info("{}", msg);
        }
        std::cout<< std::endl << std::endl;
    }
    show_console_cursor(true);
    {
        const auto msg = "所有文件拷贝完成!";
        std::cout << msg << std::endl;
        Info("{}", msg);
    }
}

static long long extract_number(const std::string &s,bool &ok)
{
    static const std::regex re(R"((\d+))");
    if (std::smatch m; std::regex_search(s,m,re) && m.size() >=2)
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
        bool ok_a{false};
        bool ok_b{false};
        const auto num_a = extract_number(a,ok_a);
        const auto num_b = extract_number(b,ok_b);
        if (ok_a && ok_b) return num_a < num_b;
        if (ok_a != ok_b) return ok_a;
        return a < b;
    });
}

void Copier::collect_files()
{
    int sum{0};
    for (const auto& date_entry: std::filesystem::directory_iterator(m_copyFileInfo.src_root))
    {
        if (!date_entry.is_directory()) continue;
        const std::string date_dir_name = date_entry.path().filename().string();
        for (const auto& label_entry: std::filesystem::directory_iterator(date_entry))
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
            const std::string label_dir_rename = std::format("{}_copied",label_dir_name);
            int count{0};
            std::vector<std::string> file_names;
            for (const auto& file_entry: std::filesystem::directory_iterator(label_entry))
            {
                if (!file_entry.is_regular_file()) continue;
                file_names.emplace_back(file_entry.path().filename().string());
                ++count;
            }
            sum += count;
            const std::filesystem::path src_last_dir = m_copyFileInfo.src_root / date_dir_name / label_dir_name;
            const std::filesystem::path src_last_rename_dir = m_copyFileInfo.src_root / date_dir_name / label_dir_rename;
            const std::filesystem::path dst_last_dir = m_copyFileInfo.dst_root / date_dir_name / label_dir_name;
            sort_files(file_names);
            m_copyFileInfo.dir_files.emplace_back(DirFiles{count,label_name,src_last_dir,src_last_rename_dir, dst_last_dir,std::move(file_names)});
        }
    }
    m_copyFileInfo.count = sum;
}


auto Copier::load_extract_params(const std::string_view label_name)->bool
{
    if (auto it = m_extract_params.find(label_name.data()); it != m_extract_params.end()) {
        return true;
    }
    std::filesystem::path extract_config_path(m_config.extract_config_dir);
    extract_config_path /= std::format("{}.json", label_name.data());

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
