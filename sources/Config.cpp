#include "../includes/Config.h"
#include "../includes/Mutex.h"
#include <memory>

ApplicationConfig::ApplicationConfig() {}

ApplicationConfig::~ApplicationConfig() {}

std::string ApplicationConfig::getLogFile() const {
  time_t now = time(0);
  tm *ltm = localtime(&now);
  auto y = 1900 + ltm->tm_year;
  auto M = 1 + ltm->tm_mon;
  auto d = ltm->tm_mday;
  auto h = 5 + ltm->tm_hour;
  auto m = 30 + ltm->tm_min;
  auto s = ltm->tm_sec;
  return std::string("Axomae_log-") + std::to_string(y) + "-" + std::to_string(M) + "-" + std::to_string(d) + "-" + std::to_string(h) + "_" + std::to_string(m) + "_" + std::to_string(s);
}

void ApplicationConfig::setConfig(const std::string &param_string) {
  is_config_init = true;
}

LoggerConfigDataStruct ApplicationConfig::generateLoggerConfigDataStruct() const {
  // TODO : generate from config file or command line options
  if (!is_config_init) {
    LoggerConfigDataStruct data;
    data.log_filters = "";
    data.log_level = LogLevel::INFO;
    std::shared_ptr<std::ostream> out(&std::cout, [](std::ostream *) {});
    data.write_destination = out;
    return data;
  } else {
    LoggerConfigDataStruct data;
    data.log_filters = "";
    data.log_level = LogLevel::INFO;
    std::shared_ptr<std::ostream> out(&std::cout, [](std::ostream *) {});
    data.write_destination = out;
    return data;
  }
}