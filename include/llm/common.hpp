#pragma once
#include <source_location>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstdint>
#include <string_view>

namespace llm
{
    const char *version();

    [[noreturn]] inline void panic(std::string_view msg, std::source_location location = std::source_location::current())
    {
        std::fprintf(stderr, "\n[llm:PANIC] %s:%u in %s\n       %.*s\n", location.file_name(), location.line(), location.function_name(), static_cast<int>(msg.size()), msg.data());
        std::abort();
    }

    enum class LogLevel : int
    {
        Trace = 0,
        Debug,
        Info,
        Warn,
        Error
    };

    inline LogLevel log_level = LogLevel::Info;

    inline void set_log_level(LogLevel level)
    {
        log_level = level;
    }

    inline const char *level_name(LogLevel l)
    {
        switch (l)
        {
        case LogLevel::Trace:
            return "TRACE";
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warn:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
        }
        return "?";
    }

    inline void log_message(LogLevel level, std::string_view msg)
    {
        if (level < log_level)
            return;
        std::fprintf(stderr, "[llm:%s] %.*s\n",
                     level_name(level),
                     static_cast<int>(msg.size()), msg.data());
    }
}

#if defined(LLM_ENABLE_ASSERTS)
#define LLM_ASSERT(cond, msg)  \
    do                         \
    {                          \
        if (!(cond))           \
            ::llm::panic(msg); \
    } while (0)
#else
#define LLM_ASSERT(cond, msg) \
    do                        \
    {                         \
        (void)sizeof(cond);   \
    } while (0)
#endif

#define LLM_CHECK(cond, msg)   \
    do                         \
    {                          \
        if (!(cond))           \
            ::llm::panic(msg); \
    } while (0)

#define LLM_UNREACHABLE() ::llm::panic("unreachable code reached")

#define LLM_LOG_TRACE(msg) ::llm::log_message(::llm::LogLevel::Trace, msg)
#define LLM_LOG_DEBUG(msg) ::llm::log_message(::llm::LogLevel::Debug, msg)
#define LLM_LOG_INFO(msg)  ::llm::log_message(::llm::LogLevel::Info, msg)
#define LLM_LOG_WARN(msg)  ::llm::log_message(::llm::LogLevel::Warn, msg)
#define LLM_LOG_ERROR(msg) ::llm::log_message(::llm::LogLevel::Error, msg)