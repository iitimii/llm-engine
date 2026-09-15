#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>

#define TEST(test_name)                                                \
    void test_name();                                                  \
    ::llmtest::Registrar registrar_##test_name(#test_name, test_name); \
    void test_name()

#define CHECK(cond)                                               \
    do                                                            \
    {                                                             \
        ++::llmtest::stats().checks;                              \
        if (!(cond))                                              \
            ::llmtest::report_failure(__FILE__, __LINE__, #cond); \
    } while (0)

#define REQUIRE(cond)                                             \
    do                                                            \
    {                                                             \
        ++::llmtest::stats().checks;                              \
        if (!(cond))                                              \
        {                                                         \
            ::llmtest::report_failure(__FILE__, __LINE__, #cond); \
            throw ::llmtest::RequireFailed{};                     \
        }                                                         \
    } while (0)

#define CHECK_CLOSE(a, b, tol)                                                             \
    do                                                                                     \
    {                                                                                      \
        ++::llmtest::stats().checks;                                                       \
        const double _a = (a), _b = (b), _t = (tol);                                       \
        if (!(std::fabs(_a - _b) <= _t + _t * std::fabs(_b)))                              \
            ::llmtest::report_failure(__FILE__, __LINE__, #a " ~= " #b " (tol " #tol ")"); \
    } while (0)

namespace llmtest
{

    struct TestCase
    {
        std::string name;
        std::function<void()> fn;
    };

    inline std::vector<TestCase> &registry()
    {
        static std::vector<TestCase> r;
        return r;
    }

    class Registrar
    {
    public:
        Registrar(std::string name, std::function<void()> fn)
        {
            registry().push_back({std::move(name), std::move(fn)});
        }
    };

    struct Stats
    {
        int checks{0};
        int check_failures{0};
    };
    inline Stats &stats()
    {
        static Stats s;
        return s;
    }

    struct RequireFailed
    {
    };

    inline void report_failure(const char *file, int line, const std::string &what)
    {
        ++stats().check_failures;
        std::fprintf(stderr, " FAIL %s:%d %s\n", file, line, what.c_str());
    }

    inline int run_all()
    {
        int tests_failed = 0;

        for (const auto &tc : registry())
        {
            const int before = stats().check_failures;

            std::printf("[ RUN ] %s\n", tc.name.c_str());
            std::fflush(stdout);

            try
            {
                tc.fn();
            }
            catch (const RequireFailed &)
            {
                // failure already recorded; just stop this test
            }
            catch (const std::exception &e)
            {
                report_failure(
                    "<exception>",
                    0,
                    std::string("uncaught: ") + e.what());
            }
            catch (...)
            {
                report_failure(
                    "<exception>",
                    0,
                    "uncaught non-std exception");
            }

            if (stats().check_failures > before)
            {
                std::printf("[ FAIL ] %s\n", tc.name.c_str());
                std::fflush(stdout);
                ++tests_failed;
            }
            else
            {
                std::printf("[ OK ] %s\n", tc.name.c_str());
                std::fflush(stdout);
            }
        }

        std::printf(
            "\n%zu test(s) run, %d failed | %d checks, %d failed\n",
            registry().size(),
            tests_failed,
            stats().checks,
            stats().check_failures);

        return tests_failed == 0 ? 0 : 1;
    }
}