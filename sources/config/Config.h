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

  /* editor */
  void setEditorLaunched(bool enable) { launches_editor = enable; }
  void setUvEditorNormalProjectionMode(bool tangent_space_mode) { uv_editor_normals_tangent_space = tangent_space_mode; }
  void setUvEditorResolutionWidth(unsigned int resolution) { uv_editor_resolution_width = resolution; }
  void setUvEditorResolutionHeight(unsigned int resolution) { uv_editor_resolution_height = resolution; }
  [[nodiscard]] bool getGuiState() { return launches_editor; }
  [[nodiscard]] int getUvEditorResolutionWidth() { return uv_editor_resolution_width; }
  [[nodiscard]] int getUvEditorResolutionHeight() { return uv_editor_resolution_height; }
  [[nodiscard]] bool isUvEditorTangentSpace() { return uv_editor_normals_tangent_space; }

  /* GPU */
  void setGpu(bool enable) { use_gpu = enable; }
  [[nodiscard]] bool usingGpu() { return use_gpu; }

  /* Threading */
  void setThreadsSize(unsigned size) { number_of_threads = size; }
  [[nodiscard]] bool isMultithreaded() { return is_multithreaded; }
  [[nodiscard]] unsigned int getThreadsNumber() { return number_of_threads; }

  /* Logging */
  void loggerSetState(bool enable) { logger_conf.enable_logging = enable; }
  [[nodiscard]] std::string getLogFile() const;
  [[nodiscard]] LoggerConfigDataStruct generateDefaultLoggerConfigDataStruct() const;
  [[nodiscard]] LoggerConfigDataStruct generateLoggerConfigDataStruct() const;

 private:
  bool is_config_init{};

  /* editor options */
  bool launches_editor{};
  bool uv_editor_normals_tangent_space{};
  unsigned int uv_editor_resolution_width;
  unsigned int uv_editor_resolution_height;

  /* GPU */
  bool use_gpu{};

  /* Threading */
  bool is_multithreaded{false};
  unsigned int number_of_threads{};

  /* Logging */
  LoggerConfigDataStruct logger_conf{};
};

#endif
