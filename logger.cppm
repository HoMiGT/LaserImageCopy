//
// Created by houmin on 2026/1/16.
//
module;

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>
#include <spdlog/async.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

export module logger;

export {
    template <typename ... Args>
    void Trace(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename ... Args>
    void Debug(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename ... Args>
    void Info(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename ... Args>
    void Warn(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename ... Args>
    void Error(fmt::format_string<Args...> fmt, Args&&... args);

    template <typename ... Args>
    void Critical(fmt::format_string<Args...> fmt, Args&&... args);
}


export class Logger
{
public:
    enum class LogLevel
    {
        Trace,
        Debug,
        Info,
        Warn,
        Error,
        Critical
    };

    static void initialize(std::string_view log_dir="logs", std::string_view app_name="LaserImageCopy",
        size_t max_file_size = 50 * 1024 * 1024, // 50 MB
        size_t max_files = 5);

    static void shutdown();

    template<typename... Args>
    static void trace(fmt::format_string<Args...> fmt, Args&&... args);

    template<typename... Args>
    static void debug(fmt::format_string<Args...> fmt, Args&&... args);

    template<typename... Args>
    static void info(fmt::format_string<Args...> fmt, Args&&... args);

    template<typename... Args>
    static void warn(fmt::format_string<Args...> fmt, Args&&... args);

    template<typename... Args>
    static void error(fmt::format_string<Args...> fmt, Args&&... args);

    template<typename... Args>
    static void critical(fmt::format_string<Args...> fmt, Args&&... args);

    static void set_level(LogLevel level);

    static void flush();

private:
    static std::shared_ptr<spdlog::logger> m_logger;
};

std::shared_ptr<spdlog::logger> Logger::m_logger{nullptr};

template <typename ... Args>
void Trace(fmt::format_string<Args...> fmt, Args&&... args)
{
    Logger::trace(fmt,std::forward<Args>(args)...);
}

template <typename ... Args>
void Debug(fmt::format_string<Args...> fmt, Args&&... args)
{
    Logger::debug(fmt,std::forward<Args>(args)...);
}

template <typename ... Args>
void Info(fmt::format_string<Args...> fmt, Args&&... args)
{
    Logger::info(fmt,std::forward<Args>(args)...);
}

template <typename ... Args>
void Warn(fmt::format_string<Args...> fmt, Args&&... args)
{
    Logger::warn(fmt,std::forward<Args>(args)...);
}

template <typename ... Args>
void Error(fmt::format_string<Args...> fmt, Args&&... args)
{
    Logger::error(fmt,std::forward<Args>(args)...);
}

template <typename ... Args>
void Critical(fmt::format_string<Args...> fmt, Args&&... args)
{
    Logger::critical(fmt,std::forward<Args>(args)...);
}

void Logger::initialize(std::string_view log_dir, std::string_view app_name, size_t max_file_size, size_t max_files)
{
    namespace fs = std::filesystem;
    try
    {
        if (const fs::path log_dir_path{log_dir}; !fs::exists(log_dir_path))
        {
            fs::create_directories(log_dir_path);
        }
        spdlog::init_thread_pool(8192,1);
        std::vector<spdlog::sink_ptr> sinks;
        sinks.reserve(3);

        // auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        // console_sink->set_level(spdlog::level::debug);
        // console_sink->set_pattern("[%^%l%$] %v");
        // sinks.emplace_back(console_sink);

        std::string detailed_log_path = std::format("{}/{}_detailed.log",log_dir,app_name);
        auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(detailed_log_path,max_file_size,max_files);
        rotating_sink->set_level(spdlog::level::info);
        sinks.emplace_back(rotating_sink);

        std::string error_log_path = std::format("{}/{}_error.log", log_dir,app_name);
        auto error_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(error_log_path,max_file_size,max_files);
        error_sink->set_level(spdlog::level::err);
        sinks.emplace_back(error_sink);

        std::string daily_log_path = std::format("{}/{}_daily.log", log_dir,app_name);
        auto daily_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(daily_log_path,0,0);
        daily_sink->set_level(spdlog::level::debug);
        sinks.emplace_back(daily_sink);

        m_logger = std::make_shared<spdlog::async_logger>(std::string(app_name),sinks.begin(),sinks.end(),spdlog::thread_pool(),spdlog::async_overflow_policy::block);
        m_logger->set_pattern("[%Y-%m-%d %H:%M:%S. %e] [%n] [%^%-8l%$] [%t] %v");
        m_logger->set_level(spdlog::level::debug);
        spdlog::set_default_logger(m_logger);


        const auto now = std::chrono::system_clock::now();
        const auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", now);
        info("========================================");
        info("日志系统初始化成功");
        info("应用程序: {}", app_name);
        info("日志目录: {}", log_dir);
        info("最大文件大小: {} MB", max_file_size / (1024 * 1024));
        info("最大文件数: {}", max_files);
        info("启动时间: {}", time_str);
        info("========================================");

    }catch (std::exception& e)
    {
        std::cerr << "日期初始化失败: " << e.what() << std::endl;
    }
}

void Logger::shutdown()
{
    if (m_logger)
    {
        const auto now = std::chrono::system_clock::now();
        const auto time_str = std::format("{:%Y-%m-%d %H:%M:%S}", now);
        info("========================================");
        info("关闭日志系统");
        info("结束时间: {}", time_str);
        info("========================================");
        m_logger->flush();
        spdlog::shutdown();
        m_logger = nullptr;

    }
}

template <typename ... Args>
void Logger::trace(fmt::format_string<Args...> fmt, Args&&... args)
{
    if (m_logger)
    {
        m_logger->trace(fmt,std::forward<Args>(args)...);
    }
}

template <typename ... Args>
void Logger::debug(fmt::format_string<Args...> fmt, Args&&... args)
{
    if (m_logger)
    {
        m_logger->debug(fmt,std::forward<Args>(args)...);
    }
}

template <typename ... Args>
void Logger::info(fmt::format_string<Args...> fmt, Args&&... args)
{
    if (m_logger)
    {
        m_logger->info(fmt,std::forward<Args>(args)...);
    }
}

template <typename ... Args>
void Logger::warn(fmt::format_string<Args...> fmt, Args&&... args)
{
    if (m_logger)
    {
        m_logger->warn(fmt,std::forward<Args>(args)...);
    }
}

template <typename ... Args>
void Logger::error(fmt::format_string<Args...> fmt, Args&&... args)
{
    if (m_logger)
    {
        m_logger->error(fmt,std::forward<Args>(args)...);
    }
}

template <typename ... Args>
void Logger::critical(fmt::format_string<Args...> fmt, Args&&... args)
{
    if (m_logger)
    {
        m_logger->critical(fmt,std::forward<Args>(args)...);
    }
}

void Logger::set_level(const LogLevel level)
{
    if (m_logger) {
        switch (level) {
        case LogLevel::Trace:
            m_logger->set_level(spdlog::level::trace);
            break;
        case LogLevel::Debug:
            m_logger->set_level(spdlog::level::debug);
            break;
        case LogLevel::Info:
            m_logger->set_level(spdlog::level::info);
            break;
        case LogLevel::Warn:
            m_logger->set_level(spdlog::level:: warn);
            break;
        case LogLevel::Error:
            m_logger->set_level(spdlog::level::err);
            break;
        case LogLevel::Critical:
            m_logger->set_level(spdlog::level::critical);
            break;
        }
    }
}

void Logger::flush()
{
    if (m_logger) {
        m_logger->flush();
    }
}
