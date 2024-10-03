#include "Config.h"
#include <iostream>
#include <memory>

ApplicationConfig::ApplicationConfig() {
  flag = 0;
  logger_conf = generateLoggerConfigDataStruct();
}
ApplicationConfig::~ApplicationConfig() = default;

std::string ApplicationConfig::getLogFile() const {
  time_t now = time(0);
  tm *ltm = localtime(&now);
  auto y = 1900 + ltm->tm_year;
  auto M = 1 + ltm->tm_mon;
  auto d = ltm->tm_mday;
  auto h = 5 + ltm->tm_hour;
  auto m = 30 + ltm->tm_min;
  auto s = ltm->tm_sec;
  return std::string("Axomae_log-") + std::to_string(y) + "-" + std::to_string(M) + "-" + std::to_string(d) + "-" + std::to_string(h) + "_" +
         std::to_string(m) + "_" + std::to_string(s);
}

LoggerConfigDataStruct ApplicationConfig::generateDefaultLoggerConfigDataStruct() const {
  LoggerConfigDataStruct data;
  data.log_filters = "";
  data.log_level = LogLevel::INFO;
  std::shared_ptr<std::ostream> out(&std::cout, [](std::ostream *) {});
  data.write_destination = out;
  data.enable_logging = false;
  return data;
}

LoggerConfigDataStruct ApplicationConfig::generateLoggerConfigDataStruct() const { return logger_conf; }

int ApplicationConfig::getThreadPoolSize() const {
  if (thread_pool) {
    return thread_pool->threadNumber();
  }
  return 0;
}

void ApplicationConfig::initializeThreadPool(int size) {
  if (size == 0)
    size = static_cast<int>(std::thread::hardware_concurrency());
  thread_pool = std::make_unique<threading::ThreadPool>(size);
}
