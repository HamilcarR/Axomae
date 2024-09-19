#include "PerformanceLogger.h"

void PerformanceLogger::startTimer() { start = std::chrono::high_resolution_clock::now(); }
void PerformanceLogger::endTimer() { end = std::chrono::high_resolution_clock::now(); }

inline void log(const std::string &str) { LOGS(str); }

void PerformanceLogger::print(TIMER type) const {
  switch (type) {
    case MILLISECONDS: {
      std::chrono::milliseconds dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      log("Duration : " + std::to_string(dur.count()) + " ms.");
    } break;

    case MICROSECONDS: {
      std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      log("Duration : " + std::to_string(us.count()) + " us.");
    } break;

    case NANOSECONDS: {
      std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      log("Duration : " + std::to_string(ns.count()) + " ns.");
    } break;
  }
}

long PerformanceLogger::getDuration() { return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); }
