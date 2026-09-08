#include"test_framework.hpp"
#include"llm/common.hpp"


TEST(check_passes_silently_on_true) {
LLM_CHECK(2 + 2 == 4, "arithmetic still works");
CHECK(true);
}

TEST(logger_respects_threshold) {
llm::set_log_level(llm::LogLevel::Warn);
LLM_LOG_INFO("this should be suppressed"); 
LLM_LOG_WARN("this should appear"); 
CHECK(true);
llm::set_log_level(llm::LogLevel::Info);
}

// TEST(abort_faliure)
// {
//     LLM_CHECK(false, "check abort");
//     // llm::set_log_level(llm::LogLevel::Debug);
//     // LLM_ASSERT(false, "assert abort");
// }
