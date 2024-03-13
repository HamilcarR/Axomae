#ifndef CONFIG_H
#define CONFIG_H
#include "Logger.h"
#include "constants.h"

/**
 * @brief File implementing a utility returning configurations states for the application , from the configuration file.
 * @file Config.h
 */

enum ConfigFlags {
  /* Editor options */
  CONF_USE_EDITOR = 1 << 0,

  CONF_UV_TSPACE = 1 << 1,
  CONF_UV_OSPACE = 1 << 2,

  /* GPU */
  CONF_USE_CUDA = 1 << 10,

  /* Threading */
  CONF_USE_MTHREAD = 1 << 20,

  /* Logging */
  CONF_ENABLE_LOGS = 1 << 30,
};

class ApplicationConfig {
 public:
  ApplicationConfig();

  void setUvEditorResolutionWidth(unsigned int resolution) { uv_editor_resolution_width = resolution; }

  void setUvEditorResolutionHeight(unsigned int resolution) { uv_editor_resolution_height = resolution; }

  [[nodiscard]] int getUvEditorResolutionWidth() { return uv_editor_resolution_width; }

  [[nodiscard]] int getUvEditorResolutionHeight() { return uv_editor_resolution_height; }

  /* Threading */
  void setThreadsSize(unsigned size) { number_of_threads = size; }

  [[nodiscard]] unsigned int getThreadsNumber() { return number_of_threads; }

  [[nodiscard]] std::string getLogFile() const;

  [[nodiscard]] LoggerConfigDataStruct generateDefaultLoggerConfigDataStruct() const;

  [[nodiscard]] LoggerConfigDataStruct generateLoggerConfigDataStruct() const;

 public:
  int flag;

 private:
  /* editor options */
  unsigned int uv_editor_resolution_width;
  unsigned int uv_editor_resolution_height;

  /* Threading */
  unsigned int number_of_threads{};

  /* Logging */
  LoggerConfigDataStruct logger_conf{};
};

#endif
