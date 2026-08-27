#pragma once

#include <cstdio>
#include <cstdlib>

namespace test {

inline int failures = 0;
inline int checks = 0;

inline void report(bool ok, const char* expr, const char* file, int line) {
  ++checks;
  if (ok) return;
  ++failures;
  std::fprintf(stderr, "FAILED %s:%d: %s\n", file, line, expr);
}

inline int finish(const char* suite) {
  std::printf("%s: %d checks, %d failures\n", suite, checks, failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}

#define CHECK(expr) ::test::report(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
