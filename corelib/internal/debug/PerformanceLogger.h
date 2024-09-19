#ifndef PERFORMANCELOGGER_H
#define PERFORMANCELOGGER_H
#include "Logger.h"
#include <iostream>
#include <string>
/**
 * @file PerformanceLogger.h
 * This file implements a class that can measure time differences
 *
 */

class PerformanceLogger {
 public:
  enum TIMER { MILLISECONDS, MICROSECONDS, NANOSECONDS };

 protected:
  std::chrono::high_resolution_clock::time_point start{};
  std::chrono::high_resolution_clock::time_point end{};
  std::chrono::milliseconds duration{};

 public:
  PerformanceLogger() = default;
  void startTimer();
  void endTimer();
  void print(TIMER type) const;
  long getDuration();
};

#define AXOMAE_START_TIMER(var_name) \
  PerformanceLogger var_name; \
  var_name.startTimer();

#define AXOMAE_END_TIMER(var_name, timer_type) \
  var_name.endTimer(); \
  var_name.print(timer_type);

#endif
