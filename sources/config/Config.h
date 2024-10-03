#ifndef CONFIG_H
#define CONFIG_H
#include "internal/debug/Logger.h"
#include "internal/thread/worker/ThreadPool.h"
/**
 * @brief File implementing a utility returning configurations states for the application , either from the configuration file or the CLI tool.
 * @file Config.h
 */

enum ConfigFlags {
  /* Editor options */
  CONF_USE_EDITOR = 1 << 0,

  /* UV editor */
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
  int flag;

 private:
  /* editor options */
  unsigned int uv_editor_resolution_width;
  unsigned int uv_editor_resolution_height;
  /* Threading */
  std::unique_ptr<threading::ThreadPool> thread_pool{};
  /* Logging */
  LoggerConfigDataStruct logger_conf{};

 public:
  ApplicationConfig();
  ~ApplicationConfig();
  ApplicationConfig(const ApplicationConfig &copy) = delete;
  ApplicationConfig(ApplicationConfig &&move) noexcept = default;
  ApplicationConfig &operator=(ApplicationConfig &&move) noexcept = default;
  ApplicationConfig &operator=(const ApplicationConfig &move) = delete;

  void setUvEditorResolutionWidth(unsigned int resolution) { uv_editor_resolution_width = resolution; }
  void setUvEditorResolutionHeight(unsigned int resolution) { uv_editor_resolution_height = resolution; }
  [[nodiscard]] int getUvEditorResolutionWidth() const { return static_cast<int>(uv_editor_resolution_width); }
  [[nodiscard]] int getUvEditorResolutionHeight() const { return static_cast<int>(uv_editor_resolution_height); }
  [[nodiscard]] int getThreadPoolSize() const;
  void initializeThreadPool(int size = 0);
  [[nodiscard]] threading::ThreadPool *getThreadPool() const { return thread_pool.get(); }
  [[nodiscard]] std::string getLogFile() const;
  [[nodiscard]] LoggerConfigDataStruct generateDefaultLoggerConfigDataStruct() const;
  [[nodiscard]] LoggerConfigDataStruct generateLoggerConfigDataStruct() const;
};

#endif
