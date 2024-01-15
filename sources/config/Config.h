#ifndef CONFIG_H
#define CONFIG_H
#include "Logger.h"
#include "constants.h"

/**
 * @brief File implementing a utility returning configurations states for the application , from the configuration file.
 * @file Config.h
 */

//?see : TODO [AX-27] Create a class reading a config file
class ApplicationConfig {
 public:
  ApplicationConfig();
  virtual ~ApplicationConfig();
  void setConfig(const std::string &param_string);
  [[nodiscard]] std::string getLogFile() const;
  [[nodiscard]] LoggerConfigDataStruct generateLoggerConfigDataStruct() const;

 private:
  bool is_config_init;
};

#endif