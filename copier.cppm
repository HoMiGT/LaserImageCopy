//
// Created by houmin on 2026/1/16.
//
module;
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <nlohmann/json.hpp>

export module copier;

import logger;

/**
 * @brief 配置结构体
 * @param src_dir 源目录
 * @param dst_dir 目标目录
 */
struct Config
{
    std::string src_dir;
    std::string dst_dir;
};

/**
 * @brief 加载配置文件
 * @param config 配置结构体引用
 * @param path 配置文件路径，默认"./config.json"
 */
void load_config(Config& config, const std::string_view path = "./config.json")
{
    std::ifstream ifs(path.data());
    if (!ifs.is_open())
    {
        const auto msg = std::format("Failed to open config file: {}",path);
        Error("{}", msg);
        throw std::runtime_error(msg);
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
        throw std::runtime_error(msg);
    }

    if (j.contains("dstDir")&&j["dstDir"].is_string())
    {
        config.dst_dir = j["dstDir"].get<std::string>();
    }else
    {
        const auto msg = std::string("Missing or incorrect fields in configuration: dst_dir");
        Error("{}",msg);
        throw std::runtime_error(msg);
    }
}

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

    void collect_files();
};


Copier::Copier()
{
    load_config(m_config);
    m_threadCount = std::thread::hardware_concurrency();
    m_copyFileInfo.src_root = std::filesystem::path(m_config.src_dir);
    m_copyFileInfo.dst_root = std::filesystem::path(m_config.dst_dir);

}

void Copier::copy()
{
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
        std::cout<< "<============== [ Task-" << task_index <<" 开始拷贝... ] ==============>" << std::endl;
        std::cout<< "标签名称: " << label_name << ", 文件数量: " << count << std::endl;
        std::cout<< "源目录: " << src_last_dir.string() << std::endl;
        // std::cout<< "重命名源目录: " << src_last_rename_dir.string() << std::endl;
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
        for (const auto& file_name: file_names)
        {
            const auto src_absolute_path = src_last_dir / file_name;
            const auto dst_absolute_path = dst_last_dir / file_name;
            if (const auto dst_absolute_parent_path = dst_absolute_path.parent_path();
                !std::filesystem::exists(dst_absolute_parent_path))
            {
                std::filesystem::create_directories(dst_absolute_parent_path);
            }
            std::filesystem::copy_file(src_absolute_path,dst_absolute_path,std::filesystem::copy_options::overwrite_existing);
            bar.set_option(option::PostfixText{std::format("{}/{}",index++, count)});
            bar.tick();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
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
    std::ranges::sort(files, [](const std::string& a, const std::string& b)
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
            std::stringstream label_name_stream;
            for (const char ch: label_dir_name)
            {
                if (ch == '_') break;
                label_name_stream << ch;
            }
            const std::string label_name = label_name_stream.str();
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
