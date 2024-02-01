#ifndef CONFIG_H
#define CONFIG_H
#include "Logger.h"
#include "constants.h"

/**
 * @brief File implementing a utility returning configurations states for the application , from the configuration file.
 * @file Config.h
 */

class ApplicationConfig {
 public:
  ApplicationConfig();

  [[nodiscard]] std::string getLogFile() const;
  [[nodiscard]] bool getGuiState() { return launch_gui; }
  [[nodiscard]] bool usingGpu() { return use_gpu; }
  [[nodiscard]] bool isMultithreads() { return multithreaded; }
  void loggerSetState(bool enable) { logger_conf.enable_logging = enable; }
  void setGuiLaunched(bool enable) { launch_gui = enable; }
  void setGpu(bool enable) { use_gpu = enable; }
  void setThreadsSize(unsigned size) { threads = size; }
  [[nodiscard]] LoggerConfigDataStruct generateDefaultLoggerConfigDataStruct() const;
  [[nodiscard]] LoggerConfigDataStruct generateLoggerConfigDataStruct() const;

 private:
  bool is_config_init{};
  bool launch_gui{};
  bool use_gpu{};
  bool multithreaded;
  unsigned int threads;

  LoggerConfigDataStruct logger_conf;
};

#endif