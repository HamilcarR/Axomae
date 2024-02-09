#include "Config.h"
#include "Mutex.h"
#include <memory>

const unsigned UV_EDITOR_RESOLUTION_W = 900;
const unsigned UV_EDITOR_RESOLUTION_H = 900;

ApplicationConfig::ApplicationConfig() {
  logger_conf = generateLoggerConfigDataStruct();
  uv_editor_resolution_width = UV_EDITOR_RESOLUTION_W;
  uv_editor_resolution_height = UV_EDITOR_RESOLUTION_H;
  uv_editor_normals_tangent_space = true;
}

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