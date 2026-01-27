//
// Created by houmin on 2026/1/16.
//
module;
#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <optional>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <thread>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

export module copier;

import logger;
import detecter;

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
Config load_config(const std::string_view path = "./config.json")
{
    Config config;
    std::ifstream ifs(path.data());
    if (!ifs.is_open())
    {
        const auto msg = std::format("Failed to open config file: {}",path.data());
        Error("{}", msg);
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
* @param label 整个标签区域
* @param euclidian_distance 欧式距离
*/
struct LabelCoordinates {
    cv::Rect qr;
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
    unsigned int m_threadCount{0};
    CopyDirInfo m_copyFileInfo;
    Recognize m_recognize{};
    Location m_location;
    bool is_ok{ false };
    std::unordered_map<std::string, ExtractParam> m_extract_params;
    float m_dpi{ 0.0f };

    void collect_files();
    bool crop(const std::filesystem::path& src_path, const std::filesystem::path& dst_path, 
        cv::Mat& mat_end, std::string_view label_name, bool is_only_end = false);
    auto load_extract_params(std::string_view label_name)->bool;

    auto find_first_second_qualified(const cv::Mat& stitch, const std::string_view setting_name,
        const int width, const int height, LabelCoordinates& left_top_label, LabelCoordinates& second_line_label)->bool;
    auto slide_capture(const cv::Mat& stitch, const LabelCoordinates& first_line_label, const LabelCoordinates& second_line_label,
        const std::filesystem::path& src_path, const std::filesystem::path& dst_path, const std::string_view setting_name,
        cv::Mat& mat_end, const bool is_only_end)->bool;


};

Copier::Copier()
    :m_config{load_config()}
    ,m_location{m_config.model}
{
    if (m_config.src_dir.empty())
    {
		Error("配置文件没有加载成功！");
        return;
    }
    m_threadCount = std::thread::hardware_concurrency();
    m_copyFileInfo.src_root = std::filesystem::path(m_config.src_dir);
    m_copyFileInfo.dst_root = std::filesystem::path(m_config.dst_dir);
    if (m_location.build()) {
        is_ok = true;
    }
}

void Copier::copy()
{
    if (!is_ok)
    {
        std::cout<< "拷贝器没有初始化成功，无法进行拷贝操作!" << std::endl;
        return;
	}
    collect_files();
    std::cout<< "拷贝的任务总数: " << m_copyFileInfo.dir_files.size() << std::endl;
    std::cout<< "拷贝文件的数量: " << m_copyFileInfo.count << std::endl;
    std::cout<< "拷贝目录的源文件根目录: " << m_copyFileInfo.src_root.string() << std::endl;
    std::cout<< "拷贝目录的目标文件根目录: " << m_copyFileInfo.dst_root.string() << std::endl;
    if (m_copyFileInfo.dir_files.empty())
    {
        std::cout<< "没有要拷贝的图片数据!" << std::endl;
        return;
    }
    using namespace indicators;
    show_console_cursor(false);
    std::cout<< std::endl << std::endl;
    int task_index{1};
    for (const auto& [count, label_name, src_last_dir,
        src_last_rename_dir, dst_last_dir, file_names]: m_copyFileInfo.dir_files)
    {
        if (!load_extract_params(label_name)) {
            std::cerr << "加载标签[" << label_name << "]的配置失败!" << std::endl;
            Warn("加载标签[ {} ]的配置失败!", label_name);
            continue;
        }
        std::cout<< "<============== [ Task-" << task_index <<" 开始拷贝... ] ==============>" << std::endl;
        std::cout<< "标签名称: " << label_name << ", 文件数量: " << count << std::endl;
        std::cout<< "源目录: " << src_last_dir.string() << std::endl;
        std::cout<< "目标目录: " << dst_last_dir.string() << std::endl;
        BlockProgressBar bar{
            option::BarWidth{80},
            option::ForegroundColor{Color::green},
            option::ShowPercentage{true},
            option::ShowElapsedTime{true},
            option::FontStyles{std::vector{FontStyle::bold}},
            option::MaxProgress{count}
        };
        int index{0};
        const auto now = std::chrono::system_clock::now();
        const auto now_str = std::format("{:%Y-%m-%d %H:%M:%S}", now);
        std::cout<< "拷贝开始时间: " << now_str << std::endl;
        cv::Mat mat_end;
        for (const auto& file_name: file_names)
        {
            const auto src_absolute_path = src_last_dir / file_name;
            const auto dst_absolute_path = dst_last_dir / file_name;
            if (const auto dst_absolute_parent_path = dst_absolute_path.parent_path();
                !std::filesystem::exists(dst_absolute_parent_path))
            {
                std::filesystem::create_directories(dst_absolute_parent_path);
            }
            if (const auto ret = crop(src_absolute_path, dst_absolute_path, mat_end, label_name); ret) {
                bar.set_option(option::PostfixText{ std::format("{}/{}",index++, count) });
                bar.tick();
            }
        }
        // 测试先注释掉重命名操作
        // std::filesystem::rename(src_last_dir,src_last_rename_dir);
        bar.mark_as_completed();
        std::cout<< "拷贝结束时间: " << std::format("{:%Y-%m-%d %H:%M:%S}", std::chrono::system_clock::now()) << std::endl;
        std::cout<<"文件拷贝完成, 并重命名源目录: " << src_last_rename_dir.string() << std::endl;
        std::cout<<"<============== [ Task-" << task_index++ <<" 拷贝完成! ] ==============>" << std::endl;
        std::cout<< std::endl << std::endl;
    }
    show_console_cursor(true);
    std::cout<<"所有文件拷贝完成!"<<std::endl;
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

void sort_files(std::vector<std::string>& files)
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

inline bool transformer_coordinates(const cv::Rect2i& box, const QrCodeResult& qrret, 
    const ExtractParam& param, float& dpi, const int width,const int height, LabelCoordinates& lc,
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
            lc.euclidian_distance = euc_fn(width,height);
        }
        else {
            lc.euclidian_distance = euc_fn(lc.label.x, lc.label.y);
        }
        lc.qr.x = p0.x + qrret.LeftTop.x;
        lc.qr.y = p0.y + qrret.LeftTop.y;
        lc.qr.width = qr_pixel;
        lc.qr.height = qr_pixel;
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
        return true;
    }
    default:
        return false;
    }
}


// 实践证明，想要截取的准确，基本每一行都需要进行找到二维码并截取，逻辑就需要进行重新梳理
bool Copier::crop(const std::filesystem::path& src_path, const std::filesystem::path& dst_path, 
    cv::Mat& mat_end, const std::string_view label_name, const bool is_only_end)
{
    const cv::Mat src = cv::imread(src_path.string(), cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        Error("读取图片失败: {}", src_path.string());
        return false;
    }
    cv::Mat stitch;
    if (!mat_end.empty()) {
        cv::vconcat(mat_end, src, stitch);
    }
    else {
        stitch = src;
    }
    const auto width = stitch.cols;
    const auto height = stitch.rows;
    int vertical_step{ 0 };
    int horizantol_step{ 0 };
    LabelCoordinates first_line_label;
    LabelCoordinates second_line_label;
    int vertical_margin{ 0 };
    bool is_find{ false };
    int mean_height{ 0 };
    constexpr auto rate = 0.25;
    if (const auto ret = find_first_second_qualified(stitch, label_name,width,height, first_line_label, second_line_label); !ret){
        Error("图片查找合格标签失败, 原路径: {}", src_path.string());
        return false;
    }
	if (const auto ret = slide_capture(stitch, first_line_label, second_line_label, src_path, dst_path, label_name, mat_end, is_only_end); !ret){
        Error("图片滑动截取失败, 原路径: {}", src_path.string());
        return false;
    }
    return true;
    
    /*while (true) {
        auto crop_width = static_cast<int>(width * rate + horizantol_step);
        crop_width = std::clamp(crop_width, 0, width);
        auto crop_height = static_cast<int>(height * rate + vertical_step);
        crop_height = std::clamp(crop_height, 0, height);
        const auto rect = cv::Rect2i{ 0,0,crop_width, crop_height};
        cv::Mat mat_find_params = stitch(rect);
        std::vector<cv::Rect2i> boxes;
        if (auto ret = m_location.infer(mat_find_params, boxes); !ret) {
            break;
        }
        if (boxes.empty())
        {
            break;
        }
        int index{ 0 };
        std::vector<LabelCoordinates> lcs;
        lcs.reserve(boxes.size());
        for (const auto& box : boxes)
        {
            auto cropped = mat_find_params(box);
            cv::imwrite(std::format("crop_{}.png", index++), cropped);
            if (!cropped.isContinuous()) {
                cropped = cropped.clone();
            }
            QrCodeResult qrret{};
            if (const auto ret = m_recognize.detect(cropped, qrret); !ret) {
                Warn("{}", m_recognize.what());
                continue;
            }
            LabelCoordinates lc{};
            const auto is_ok = transformer_coordinates(box, qrret, m_extract_params[label_name.data()], m_dpi, crop_width, crop_height, lc);
            if (!is_ok) {
                continue;
            }
            if (lc.label.x < 0 || lc.label.y < 0 || lc.label.x > crop_width || lc.label.y > crop_height) {
                continue;
            }
            lcs.emplace_back(std::move(lc));
        }
        if (lcs.empty()) {
            break;
        }
        auto min_it = std::min_element(
            lcs.begin(), lcs.end(),
            [](const LabelCoordinates& a, const LabelCoordinates& b) {
                return a.euclidian_distance < b.euclidian_distance;
            }
        );
        const LabelCoordinates* p_min_label{ nullptr };
        const LabelCoordinates* p_next_line_label{ nullptr };
        if (min_it != lcs.end()) {
            p_min_label = &*min_it;
            for (const auto& item : lcs) {
                if (mean_height == 0) {
                    mean_height = static_cast<int>((p_min_label->label.height + item.label.height) / 2.0 + 0.5);
                }
                else {
                    mean_height = static_cast<int>((mean_height + static_cast<int>((p_min_label->label.height + item.label.height) / 2.0 + 0.5)) / 2.0 + 0.5);
                }
                const auto mean_height = static_cast<int>((p_min_label->label.height + item.label.height) / 2.0);
                const auto distance = std::abs(p_min_label->label.y - item.label.y);
                if (distance > mean_height * 0.8 && distance < mean_height * 2) {
                    p_next_line_label = &item;
                    break;
                }
            }
        }
        if (!p_min_label) {
            break;
        }
        if (!p_next_line_label) {
            if (crop_width == width && crop_height == height) {
                break;
            }
            vertical_step += p_min_label->label.height * 1.2;
            horizantol_step += p_min_label->label.width * 1.2;
            continue;
        }
        vertical_margin = std::abs(p_min_label->label.y - p_next_line_label->label.y) - mean_height;
        vertical_margin = std::clamp(vertical_margin, 0, vertical_margin);
        first_label = *p_min_label;
        is_find = true;
        cv::imwrite("frame.png", mat_find_params);
        break;
    }*/
    /*const auto count = height / first_line_label.label.height;
    int start_row{ 0 };
    int end_row{ 0 };
    const auto parent_path = dst_path.parent_path();
    const auto filename = dst_path.stem();
    const auto suffix = dst_path.extension();
    std::vector<std::filesystem::path> save_names;
    save_names.reserve(count);
    vertical_margin = 0;
    for (auto i{ 0 }; i < count; ++i) {
        start_row = first_line_label.label.y - vertical_margin + i * mean_height;
        start_row = std::clamp(start_row, 0, start_row);
        end_row = first_line_label.label.y + vertical_margin + (i + 1) * mean_height;
        if (start_row > height) {
            break;
        }
        if (end_row > height) {
            mat_end = stitch.rowRange(start_row, height).clone();
            break;
        }
        if (is_only_end) {
            continue;
        }
        const auto one_line = stitch.rowRange(start_row, end_row);
        const auto save_name = std::format("{}_{}.{}",filename.string(),i,suffix.string());
        const auto path = parent_path / save_name;
        save_names.emplace_back(path);
        cv::imwrite(path.string(),one_line);
    }
    if (is_only_end) {
        return true;
    }
    std::stringstream ss;
    for (const auto& item : save_names) {
        ss << item.string() << ",";
    }
    auto split_path = ss.str();
    if (!split_path.empty() && split_path.back() == ',') {
        split_path.pop_back();
    }
    Info("\n源文件路径: {},\n目标文件路径: {},\n拆分目标文件路径: {}", src_path.string(),dst_path.string(), split_path);
    Logger::flush();
    return true;*/
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

auto Copier::find_first_second_qualified(const cv::Mat& stitch, const std::string_view setting_name,
    const int width, const int height, LabelCoordinates& first_line_label, LabelCoordinates& second_line_label) -> bool
{
    int mean_height{ 0 };
	int vertical_step{ 0 };
    int horizantol_step{ 0 };
    int vertical_margin{ 0 };
    constexpr auto rate = 0.25;
    while (true) {
        auto crop_width = static_cast<int>(width * rate + horizantol_step);
        crop_width = std::clamp(crop_width, 0, width);
        auto crop_height = static_cast<int>(height * rate + vertical_step);
        crop_height = std::clamp(crop_height, 0, height);
        const auto rect = cv::Rect2i{ 0,0,crop_width, crop_height };
        cv::Mat mat_find_params = stitch(rect);
        std::vector<cv::Rect2i> boxes;
        if (auto ret = m_location.infer(mat_find_params, boxes); !ret) {
            continue;
        }
        if (boxes.empty())
        {
            continue;
        }
        int index{ 0 };
        std::vector<LabelCoordinates> lcs;
        lcs.reserve(boxes.size());
        for (const auto& box : boxes)
        {
            auto cropped = mat_find_params(box);
            cv::imwrite(std::format("crop_{}.png", index++), cropped);
            if (!cropped.isContinuous()) {
                cropped = cropped.clone();
            }
            QrCodeResult qrret{};
            if (const auto ret = m_recognize.detect(cropped, qrret); !ret) {
                Warn("{}", m_recognize.what());
                continue;
            }
            LabelCoordinates lc{};
            const auto is_ok = transformer_coordinates(box, qrret, m_extract_params[setting_name.data()], m_dpi, crop_width, crop_height, lc);
            if (!is_ok) {
                continue;
            }
            if (lc.label.x < 0 || lc.label.y < 0 || lc.label.x > crop_width || lc.label.y > crop_height) {
                continue;
            }
            lcs.emplace_back(std::move(lc));
        }
        if (lcs.empty()) {
            continue;
        }
        auto min_it = std::min_element(
            lcs.begin(), lcs.end(),
            [](const LabelCoordinates& a, const LabelCoordinates& b) {
                return a.euclidian_distance < b.euclidian_distance;
            }
        );
        const LabelCoordinates* p_min_label{ nullptr };
        const LabelCoordinates* p_next_line_label{ nullptr };
        if (min_it != lcs.end()) {
            p_min_label = &*min_it;
            for (const auto& item : lcs) {
                if (mean_height == 0) {
                    mean_height = static_cast<int>((p_min_label->label.height + item.label.height) / 2.0 + 0.5);
                }
                else {
                    mean_height = static_cast<int>((mean_height + static_cast<int>((p_min_label->label.height + item.label.height) / 2.0 + 0.5)) / 2.0 + 0.5);
                }
                const auto distance = std::abs(p_min_label->label.y - item.label.y);
                if (distance > mean_height * 0.8 && distance < mean_height * 2) {
                    p_next_line_label = &item;
                    break;
                }
            }
        }
        if (!p_min_label) {
            break;
        }
        if (!p_next_line_label) {
            if (crop_width == width && crop_height == height) {
                break;
            }
            vertical_step += p_min_label->label.height * 1.2;
            horizantol_step += p_min_label->label.width * 1.2;
            continue;
        }
        else {
            first_line_label = *p_min_label;
            second_line_label = *p_next_line_label;
            return true;
        }
    }   
    return false;
}

auto Copier::slide_capture(const cv::Mat& stitch, const LabelCoordinates& first_line_label,const LabelCoordinates& second_line_label,
    const std::filesystem::path& src_path, const std::filesystem::path& dst_path,const std::string_view setting_name, 
    cv::Mat& mat_end, const bool is_only_end) -> bool
{
	auto mean_height = static_cast<int>((first_line_label.label.height + second_line_label.label.height) / 2.0 + 0.5);
	auto coordinate_offset = std::abs(second_line_label.label.y - first_line_label.label.y) - mean_height;
    coordinate_offset = std::clamp(coordinate_offset, 0, coordinate_offset);
    coordinate_offset = static_cast<int>(coordinate_offset / 2.0 + 0.5);
    const auto width = stitch.cols;
	const auto height = stitch.rows;
    int vertical_step{mean_height + 2 * coordinate_offset};
	const auto count = height / mean_height;
    const auto parent_path = dst_path.parent_path();
    const auto filename = dst_path.stem();
    const auto suffix = dst_path.extension();
    std::vector<std::filesystem::path> save_names;
    save_names.reserve(count);
	auto one_line_start_row = first_line_label.label.y - coordinate_offset;
	one_line_start_row = std::clamp(one_line_start_row, 0, height);
	auto one_line_end_row = first_line_label.label.y + mean_height + coordinate_offset;
	one_line_end_row = std::clamp(one_line_end_row, 0, height);
    const auto one_line = stitch.rowRange(one_line_start_row, one_line_end_row);
    const auto one_line_save_name = std::format("{}_0{}", filename.string(), suffix.string());
    const auto one_line_path = parent_path / one_line_save_name;
    save_names.emplace_back(one_line_path);
	cv::imwrite(one_line_path.string(), one_line);
	int previous_line_y = first_line_label.label.y;
	int previous_line_height = first_line_label.label.height;
    auto start_row = first_line_label.label.y - coordinate_offset + mean_height;
    start_row = std::clamp(start_row, 0, height);
	auto end_row = first_line_label.label.y + coordinate_offset + 2 * mean_height;
    end_row = std::clamp(end_row, 0, height);
    for (auto i{ 1 }; i < count; ++i) {
        bool is_found{ false };
        int adaptive_horizontal_offset = 0;
        LabelCoordinates current_line_label;
        while (true) {
            while (true) {
                auto start_col = static_cast<int>(first_line_label.label.x - first_line_label.label.width * 0.5 + 0.5 + adaptive_horizontal_offset);
                start_col = std::clamp(start_col, 0, width);
                auto end_col = static_cast<int>(first_line_label.label.x + first_line_label.label.width * 1.5 + 0.5 + adaptive_horizontal_offset);
                end_col = std::clamp(end_col, 0, width);
                auto slide_width = end_col - start_col;
                auto slide_height = end_row - start_row;
                if (slide_width <= 0 || slide_height <= 0) {
                    break;
                }
                slide_width = start_col + slide_width > width ? start_col+slide_width - width : slide_width;
				slide_height = start_row + slide_height > height ? start_row+slide_height - height : slide_height;
                const cv::Rect slide_window{ start_col, start_row, slide_width, slide_height };
                const auto slide_mat = stitch(slide_window);
				cv::imwrite("slide_mat.png", slide_mat);
                std::vector<cv::Rect2i> boxes;
                if (auto ret = m_location.infer(slide_mat, boxes); !ret) {
                    adaptive_horizontal_offset += static_cast<int>(first_line_label.label.width * 0.3 + 0.5);
                    continue;
                }
                if (boxes.empty())
                {
                    adaptive_horizontal_offset += static_cast<int>(first_line_label.label.width * 0.3 + 0.5);
                    continue;
                }
                int index{ 0 };
                std::vector<LabelCoordinates> lcs;
                lcs.reserve(boxes.size());
                for (const auto& box : boxes)
                {
                    auto cropped = slide_mat(box);
                    if (!cropped.isContinuous()) {
                        cropped = cropped.clone();
                    }
                    QrCodeResult qrret{};
                    if (const auto ret = m_recognize.detect(cropped, qrret); !ret) {
                        Warn("{}", m_recognize.what());
                        continue;
                    }
                    LabelCoordinates lc{};
                    const auto is_ok = transformer_coordinates(box, qrret, m_extract_params[setting_name.data()], m_dpi, slide_width, slide_height, lc,start_col, start_row);
                    if (!is_ok) {
                        continue;
                    }
                    if (lc.label.x < 0 || lc.label.y < 0 || lc.label.x > slide_width + start_col || lc.label.y > slide_height + start_row) {
                        continue;
                    }
                    lcs.emplace_back(std::move(lc));
                }
                if (lcs.empty()) {
                    adaptive_horizontal_offset += static_cast<int>(first_line_label.label.width * 0.3 + 0.5);
                    continue;
                }
                auto min_it = std::min_element(
                    lcs.begin(), lcs.end(),
                    [](const LabelCoordinates& a, const LabelCoordinates& b) {
                        return a.euclidian_distance < b.euclidian_distance;
                    }
                );
                if (min_it != lcs.end()) {
                    current_line_label = *min_it;
                    is_found = true;
                    break;
                }
            }
            if (is_found) {
                break;
            }
            else {
                end_row += mean_height * 0.3;
				end_row = std::clamp(end_row, 0, height);
            }
        }
        if (is_found) {
            mean_height = static_cast<int>((previous_line_height + current_line_label.label.height) / 2.0 + 0.5);
            coordinate_offset = std::abs(current_line_label.label.y - previous_line_y) - mean_height;
            coordinate_offset = std::clamp(coordinate_offset, 0, coordinate_offset);
            coordinate_offset = static_cast<int>(coordinate_offset / 2.0 + 0.5);
            previous_line_y = current_line_label.label.y;
			previous_line_height = current_line_label.label.height;
			const auto cut_start_row = current_line_label.label.y - coordinate_offset;
			auto cut_end_row = current_line_label.label.y + mean_height + coordinate_offset;
			cut_end_row = std::clamp(cut_end_row, 0, height);
            if (!is_only_end) {
                const auto one_line = stitch.rowRange(cut_start_row, cut_end_row);
                const auto save_name = std::format("{}_{}{}", filename.string(), i, suffix.string());
                const auto path = parent_path / save_name;
                save_names.emplace_back(path);
                cv::imwrite(path.string(), one_line);
            }
            start_row = cut_end_row - coordinate_offset;
            end_row = cut_end_row + mean_height + 2 * coordinate_offset;
            if (start_row > height) {
                break;
            }
            if (end_row > height) {
				mat_end = stitch.rowRange(cut_start_row, height).clone();
                break;
            }
        }
        else {
            start_row += mean_height - coordinate_offset;
            end_row +=  mean_height + 2 * coordinate_offset;
            if (start_row > height) {
                break;
            }
            if (end_row > height) {
                break;
            }
        }
    }
    if (is_only_end) {
        return true;
    }
    if (save_names.empty()) {
        return false;
    }
    std::stringstream ss;
    for (const auto& item : save_names) {
        ss << item.string() << ",";
    }
    auto split_path = ss.str();
    if (!split_path.empty() && split_path.back() == ',') {
        split_path.pop_back();
    }
    Info("\n源文件路径: {},\n目标文件路径: {},\n拆分目标文件路径: {}", src_path.string(), dst_path.string(), split_path);
    Logger::flush();
    return true;
}
