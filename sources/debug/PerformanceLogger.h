#ifndef PERFORMANCELOGGER_H
#define PERFORMANCELOGGER_H
#include "Logger.h"

/**
 * @file PerformanceLogger.h
 * This file implements a class that can measure time differences
 *
 */

class PerformanceLogger : protected AbstractLogger {
 protected:
  std::chrono::high_resolution_clock::time_point start{};
  std::chrono::high_resolution_clock::time_point end{};
  std::chrono::milliseconds duration{};

 public:
  PerformanceLogger() = default;
  void startTimer() {
    std::cout << "Timer started!\n";
    start = std::chrono::high_resolution_clock::now();
  }
  void endTimer() {
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Timer end!\n";
  }
  void print() const override {
    std::chrono::milliseconds dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Duration : " << dur.count() << "ms\n";
  }
};

#endif